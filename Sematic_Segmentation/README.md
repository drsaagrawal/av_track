# Lyft Challenge Winner

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528697144561_e5.png)


To solve this challenge I develop a novel approach to combine modern deep learning with classical computer vision techniques to achieve highest score on leader-board at the time of the submission. I implemented a highly parallel CPU/GPU pipeline to achieve highest FPS in the top 10 contestants.

# My Approach
## Neural Network Selection

I evaluated the following methods to use with this challenge:


-  [Google’s DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)
-  [FCN-Alexnet](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn-alexnet)

I used [Cityscapes](https://www.cityscapes-dataset.com/) as common starting point.  DeepLab’s supports Cityscapes out of the box. [FCN-Alexnet](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn-alexnet) is my own customized implementation of this [paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) from [fcn.berkeleyvision.org](http://fcn.berkeleyvision.org/) in TensorFlow. I adopted it to train on [Cityscapes](https://www.cityscapes-dataset.com/) for my earlier work before this challenge. 

This challenge required us to only label three unique classes. I settled for my FCN approach. Even though FCN is not state of the art when it comes to Semantic Segmentation, I’ve learned that FCN performs reasonably well with small number of unique labels. FCN also has modest GPU requirements compared to some other state of the art approaches that makes it easier to train and faster to infer on older Nvidia K80 GPU made available for this challenge. Just for comparison [Google’s DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab) required me to use 4x Nivida P100 GPUs before it started to train reasonably on Lyft datasheet.

Please see Appendix A for details about the Experimental Framework.

## Establishing a Baseline

I implemented a preprocessing routine in Python to apply same training label to road markings and to road surface. Then I set ego car label as background. I used [Cityscpae](https://www.cityscapes-dataset.com/) compatible labels while saving ground truth files because it can work with Google DeepLabv3 with pre-trained weighs for Cityscapes. 

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528696532776_Screenshot+from+2018-06-09+02-06-04.png)


I modified my FCN Cityscapes implementation to restrict detection to :

- Road, 
- Vehicle 
- Background

I took the initial 1000 training images made available to us and trained for 40 epoch. 
The result was an Average F score 0.81 and FPS of 5. This established that the training datasheet is in correct format and there are no bugs in the implementation.

# Improving F Score

I tried classical data augmentation techniques, by randomly mirroring and jittering the images by translating them up to 32 pixels. I learned that there is not much benefit and this is also unnecessary since more training data can easily be generated from [Carla](http://carla.org/) simulator that is available freely. I noticed that more training data does improves score. Please see Appendix A. 

Eventually I end up with 10k+ images trained on 200+ epochs, with following scores:

| Car F score: 0.778          | Car Precision: 0.914  | Car Recall: 0.751  |
|-----------------------------| --------------------- | -------------------|
| Road F score: 0.987         | Road Precision: 0.998 | Road Recall: 0.944 |
| **Averaged F score: 0.883** |                       |                    |

This hinted that, to move further up the leaderboad I needed to change something else. Just ading more data and epoch are not enough.

## Cropping

Cars occupy much small area in images as compered to road. As I increase the training, the network develops a bias towards road.

An FCN for Semantic Segmentation essentially downsamples an image via pooling and striding convolution and then upsamples it. This introduces problems when trying to classify labels that occupies small area in the image. My Neural Network accepts input size 576x160. When I resize from original image dimension of 800x600, this also introduces quantization noise where small details in the image are lost. The results above show that I needed to be smart how I resize the image.

![How tiny detail don’t infer correctly](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528047972606_Screenshot+from+2018-06-03+22-45-54.png)


I noticed that there is no meaningful detail in the bottom of the images due to hood of the ego vehicle. Similarly sky has no information about cars or road. So I decided to crop my training data and inference images as follows:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528696566490_Screenshot+from+2018-06-09+15-34-42.png)


 I trained again and got better results:

| Car F score: 0.841          | Car Precision: 0.928  | Car Recall: 0.821  |
| --------------------------- | --------------------- | ------------------ |
| Road F score: 0.993         | Road Precision: 0.999 | Road Recall: 0.971 |
| **Averaged F score: 0.917** |                       |                    |

Cars F score is doing much better than the road. I recognize that cars’ relatively poor recall is pulling the overall score down.

## Deep Learning based techniques

Typically at this point I would try to address the class imbalance in the training data. Fully convolutional training can balance classes by weighting or sampling the loss. Considering the final goal of this challenge. I thought it might be faster to use classical computer vision technique as discussed below:

## Binary Dilation

The score calculation places a higher penalty if my method fails to classify pixels that belong to car. The fact that cars are much less pixels in training data and the fact that FCN introduces quantization, it struggles to label all of car pixels in fine detail. I observe that Car has a high precision score, meaning when we see a car, we do label it but only some part of it. I made another observation that the detected pixels tend to be in the center of the car. 

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528696625849_Screenshot+from+2018-06-09+22-29-42.png)


So I decided to expend the car detection pixels by classical computer vision technique called [Dilation](https://en.wikipedia.org/wiki/Dilation_(morphology)). It expands the detection area uniformly around the detection center. I perform dilation on the final predicted binary image for car only, just before encoding it to json. The below score are with dilation of 1 iteration.

| Car F score: 0.852          | Car Precision: 0.907  | Car Recall: 0.839  |
| --------------------------- | --------------------- | ------------------ |
| Road F score: 0.993         | Road Precision: 0.999 | Road Recall: 0.971 |
| **Averaged F score: 0.922** |                       |                    |

This improve Car’s Recall at the cost of reducing car’s Precision. This is ok since the scorning formula requires higher recall for car and higher precision for road. I can improve score further by making multiple iteration of Dilation operation.

**Other ideas**
Final class score from my FCN are produced by a [softmax](https://en.wikipedia.org/wiki/Softmax_function) function, which are probabilities of each pixel belonging to each class, that we care about. I feed it to an [argmax](https://en.wikipedia.org/wiki/Arg_max) function to choose the highest probability class. Next item on my list was to tweak the final class decision to have a lower threshold towards cars. Another option was to change my [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) [loss function](http://theanets.readthedocs.io/en/stable/api/losses.html) to be same as Lyft's ranking formula.
Since I'd already made it to the top at this point so I didn't try these instead I decided to invest my time in improving the run-time performance of the model.

# Improving FPS

Running on training graph give less than 5fps. I used the following techniques to improve performance

## Optimized Inference graph

First I froze the graph, which is the process of converting TensorFlow variables into constants. During inference, variables become unnecessary since the values they store don’t change. computer can work with faster with constants. Additional benefits of freezing the graph are:


- Unnecessary nodes related to training are removed
- The model can be contained entirely in one protobuf file
- Simpler graph structure
- Easier to deploy due to everything being in one file.

Then I used Tensorflow tool to optimize graph for inference which does the following:

- Removes training-specific and debug-specific nodes
- Fuses common operations
- Removes entire sections of the graph that are never reached

Using optimized graph nearly doubled the speed to ~9fps.

I also tried quantizing the graph but noticed slight reduction in performance on my laptop. So I didn’t tried it on challenge workspace. 


## Batching

I profiled GPU activity and noticed that 50% of time GPU was idle!

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528696692709_fps1.png)


Sending one frame to GPU for inference is really inefficient considering the high latency of uploading data to the GPU and fetching results. I noticed it even worse problem on GCP with virtualized GPU in sharing setup.

I re-coded my inference logic to accept a batch of images of configurable size. Sending 10~50 frames at once improve performance to more than 11fp.

When I run *nvidia-smi*, I notice patches of time when GPU is idle, so there is room for improvement.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528696708182_fps2.png)

## Parallel preprocess

I started with a naive pipeline that did not take advantage of multi-core CPU and GPU parallelism. This is evident by GPU utilization graph shown above.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528696738343_Screenshot+from+2018-06-10+18-25-49.png)


Input to the Neural network was via scikit-video that uses FFmepg under the hood. I needed to crop and resize the images before feeding to the FCN. Doing this in python when video has already bean read to memory is inefficient since it block the GPU waiting for CPU to perform these prepossessing operations. To fix this I start the FFmpeg as Python sub-process early on. I use custom FFmepg command line to crop and resize the video at the same time while reading. FFmpeg continues to decode video in parallel as I go on to initialize TensorFlow and load graph. When I am finally ready to run inference, I read the video from FFmpeg using pipe.
This bumps up fps to ~13.

*nvidia-smi*, still reports some patches of time when GPU is idle, so there still is room for improvement.

## Parallel post process.

After segmentation map has been retuend by the GPU, there are following operations that still need to be performed.


- Uncrop - by append background label to hood and the sky
- Resize
- Binary image creation for car and road
- Binary Dilation for car
- JSON encode and print

This holds up the GPU while CPU finishes up these operation. I decided to use Python threading here instead of multiprocessing because lot of the time main thread is waiting for the GPU to finish. This reduces the overhead of [GIL](https://wiki.python.org/moin/GlobalInterpreterLock). Threading also allows me to easily share numpy arrays.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528696770858_Screenshot+from+2018-06-10+18-26-27.png)


I implemented it in a way that as soon as Tensorflow session returns a batch of results, I post it to a queue. Anther thread waits on this queue and performs the above mentioned CPU bound operations in parallel while GPU goes on and processes the next batch.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528049056320_Screenshot+from+2018-06-03+23-04-05.png)


This give me a whopping 16.6 fps.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_425C5D31460EC0644CFCDAB4D644C84378A0A89401D1B7DEC6850992DBD4409F_1528696802745_fps3.png)

# Appendix A
## Optimization

I used Adam optimizer with minibatch size of 13 and learning rate of 1e-4. keep_prob was increased gradually from .5 to 1.0 during training.

## Training Data
| Source         | Link | count                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------- |
| Official         | https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz | 1000                  |
| chinkiat         | https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/                                                 | 3000                  |
| Mohamed Eltohamy | https://drive.google.com/open?id=1NimO26IH_Y8DziDMsgBCZeHlT3duj4-e                                               | 1000                  |
| phmagic          | https://www.dropbox.com/s/1etgf32uye2iy8q/world_2_w_cars.tar.gz?dl=0                                             | 2535 (after cleaning) |
|                  | My own                                                                                                           | 3000                  |

## Dataset Split

I set aside 200 images as validation set.

## L2 regularization

(1e-3) L2 regularization was added to transpose convolution layer.

# Acknowledgements
- Ong Chin-Kiat (chinkiat), Phu Nguyen (phmagic), and Mohamed Eltohamy shared extra training data from CARLA.
- Phu Nguyen (phmagic) shared a nice tip that OpenCV was faster than PIL for encoding to PNG format. 

# Appendix B - Other Dataset support
## Datasets

To switch datasets, look at the start of run() in main.py and replace helper.KittiDataset with for exmaple helper.LyftData

**Kitti**

Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip). Extract the dataset in the `data` folder. This will create the folder `data_road` with all the training a test images.

**Cityscape**

Register at [The Cityscapes Dataset](https://www.cityscapes-dataset.com/downloads/). Download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip. Extrac both to `data`folder such that you have `data/gtFine` and `datas/leftImg8bit`.

**kitti2015**

Download data_semantics.zip and provide path to the folder containing *image_2*.

**Robust Vision Challenge**

Download and setup devkit and provide path to rob_devkit/segmentation/datasets_kitti2015 it brings in additional support for following datasets.

- [WildDash](http://wilddash.cc/)
- [ScanNet](http://www.scan-net.org/)

**Run**
Run the following command to run the project:

    python main.py

**Kitti Results**
**Mean IOU: 0.960**

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/runs/1521580880.289971/um_000003.png)


 

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/runs/1521580880.289971/um_000005.png)


 

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/runs/1521580880.289971/um_000007.png)


 

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/runs/1521580880.289971/um_000013.png)


**Cityscape Results**

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/images/1521974760.190023/munich_000149_000019_leftImg8bit.png)


 

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/images/1521974760.190023/munich_000159_000019_leftImg8bit.png)


 

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/images/1521974760.190023/munich_000160_000019_leftImg8bit.png)


 

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/images/1521974760.190023/munich_000164_000019_leftImg8bit.png)


**Cross-entropy Loss**

![sample](https://github.com/asadziach/CarND-Semantic-Segmentation/raw/master/images/kitti-loss.png)


**Tips**

- The link for the frozen `VGG16` model is hardcoded into `helper.py`. The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers.


