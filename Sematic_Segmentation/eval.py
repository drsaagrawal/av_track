import tensorflow as tf
import sys, json, base64
import numpy as np
import cv2
import timeit
import scipy.ndimage
import subprocess as sp
import os
import threading
import queue


# This runs in a thread and performs CPU bound postprocessing.
def postprocess(q, answer_key, batch_size):
    
    nn_shape = (160, 576)
    crop_size = (296, 800)
    org_size = (600, 800)
    crop_offset = 226
    frame = 1    
    
    while True:
        # Read next batch
        seg_map_batch = q.get()
        outarry = seg_map_batch[0]
                
        outarry = outarry.reshape(batch_size[0], nn_shape[0], nn_shape[1])

        for i in range(batch_size[0]):
            seg_map = outarry[i, ...].astype(np.uint8)
                                           
            upscaled = cv2.resize(seg_map, dsize=(crop_size[1], crop_size[0]), interpolation=cv2.INTER_NEAREST)
            # un-crop
            padded = np.zeros(org_size, dtype=np.uint8)  # bg type
            padded[crop_offset:upscaled.shape[0] + crop_offset, :upscaled.shape[1]] = upscaled
            # ready to create binary images
            binary_car_result, binary_road_result = get_binary_image(padded)        
            answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
            # Increment frame
            frame += 1      
              
        q.task_done()


def get_decoded_vidoe(file, raw_video):
    # NN expects images cropped and resized.
    command = [ "ffmpeg",
        '-i', file,
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-filter:v',
        'crop=800:296:0:226,scale=576x160',
        '-']
    FNULL = open(os.devnull, 'w')
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=FNULL, bufsize=-1)   
    raw_video[0] = pipe.stdout.read() 

    
def get_binary_image(seg_map):
    
    road_class = 1
    car_class = 2
    gt = seg_map
    cargt = gt == car_class
    roadgt = gt == road_class
    '''
    Binary dilation improves Car’s Recall at the cost of reducing 
    car’s Precision. This is ok since the scorning formula 
    requires higher recall for car and higher precision for road.
    num_iterations is hyper-parameter. 
    '''
    cargt = scipy.ndimage.binary_dilation(cargt, iterations=3)
    
    return cargt.astype(np.uint8), roadgt.astype(np.uint8) 


# Define encoder function
def encode(array):
    _, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

    
def main():
    
    nn_shape = (160, 576)
    batch_size = [50]
    answer_key = {}
    raw_video = [np.zeros(0)]
    q = queue.Queue()
    file = sys.argv[-1]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    myname = __file__
    if file == myname:
        print ("Error loading video")
        exit()
    '''
    Start a thread that starts FFmepg subprocess and reads raw video from pipe. This can 
    happen in parallel to TensorFlow initialization and loading of model, saving seconds.
    '''    
    download_thread = threading.Thread(target=get_decoded_vidoe, args=[file, raw_video])
    download_thread.start()
    '''
    Start the post processing thread now, so that it is ready when we need it later.
    '''
    postprocess_threads = threading.Thread(target=postprocess, args=(q, answer_key, batch_size))
    postprocess_threads.daemon = True
    postprocess_threads.start()
            
    with tf.gfile.GFile('./graph_optimized.pb', 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    G = tf.Graph()

    with tf.Session(graph=G) as sess:
        logits, = tf.import_graph_def(graph_def_optimized, return_elements=['ArgMax:0'])
        image_input = G.get_tensor_by_name('import/image_input:0')
        keep_prob = G.get_tensor_by_name('import/keep_prob:0')

        sess.run(tf.global_variables_initializer())

        start_time = timeit.default_timer()
        download_thread.join()
        elapsed = timeit.default_timer() - start_time
        print("Waited for ffmpeg: ", elapsed, file=sys.stderr)
        
        video = np.fromstring(raw_video[0], dtype='uint8')
        n_frames = video.size // (nn_shape[0] * nn_shape[1] * 3)
        video = video.reshape((n_frames, nn_shape[0], nn_shape[1], 3))
        


        start_time = timeit.default_timer()
         
        if not video.shape[0] % batch_size[0] == 0:
            batch_size[0] = 1
            print("WARNING! Batching disabled.", file=sys.stderr)
            
        batch_size = batch_size[0]                
        loops_req = video.shape[0] // batch_size 
        print("Loops needed: ", loops_req, ", Batch size:", batch_size, file=sys.stderr)
              
        for c in range(loops_req):   
            start = c * batch_size
            end = start + batch_size   
            clip = video[start:end, ...]
            seg_map_batch = sess.run([logits],
                               {keep_prob: 1.0, image_input: clip})
            q.put(seg_map_batch)
        # Wait for any pending post processing to complete    
        q.join()
            
        elapsed = timeit.default_timer() - start_time  
        fps = video.shape[0] / elapsed 
        start_time = timeit.default_timer()
        # Print output in proper json format
        print (json.dumps(answer_key))
        elapsed = timeit.default_timer() - start_time
        print("json time: ", elapsed, file=sys.stderr)  
        print("Video fps: ", fps, file=sys.stderr)


if __name__ == '__main__':
    main()
