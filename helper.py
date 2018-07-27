import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cityscape_labels
import labels
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __init__(self, data_folder, image_shape):
        '''
        :param data_folder: Path to folder that contains all the datasets
        :param image_shape: Tuple - Shape of image
        '''
        pass

    @abstractmethod
    def gen_batch_function(self):
        pass

    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def save_inference_samples(self, runs_dir, sess, logits, keep_prob, input_image):
        """
        Generate test output using the test images
        :param runs_dir: Folder to place output
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        pass
    
class RvcDataset(Dataset):

    '''
    class attributes
    '''

    def __init__(self, data_folder, image_shape):
        '''
        :param data_folder: Path to folder that contains all the datasets
        :param image_shape: Tuple - Shape of image
        '''
        self.data_folder = data_folder
        self.image_shape = image_shape
        colors = {label: np.array(gt.color)
                          for label, gt in enumerate(labels.labels)}
        self.gt_colors = colors

                    
        self.gt_list = [i for i in range(73)]

        pngs = glob(os.path.join(data_folder, 'training','image_2', '*.png'))
        jpgs = glob(os.path.join(data_folder, 'training','image_2', '*.jpg'))
        self.image_paths = pngs + jpgs
        
        test_folder_name = "image_2"
        pngs = glob(os.path.join(data_folder, 'test',test_folder_name, '*.png'))
        jpgs = glob(os.path.join(data_folder, 'test',test_folder_name, '*.jpg'))
        self.test_image_paths = pngs + jpgs
                
        self.label_paths = {
            os.path.basename(path): path
            for path in glob(os.path.join(data_folder, 'semantic', '*.png'))}

    def get_num_classes(self):
        return len(self.gt_list)

    def gen_batch_function(self):
        """
        Generate function to create batches of training data
        :return:
        """
        image_paths = self.image_paths
        label_paths = self.label_paths 
        image_shape = self.image_shape

        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in image_paths[batch_i:batch_i + batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file).replace(".jpg", ".png")]

                    image = scipy.misc.imresize(
                        scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(
                        scipy.misc.imread(gt_image_file), image_shape)

                    gt_bg = np.zeros(
                        [image_shape[0], image_shape[1]], dtype=bool)
                    gt_list = []
                    for label in self.gt_list[1:]: #auto bg
                        gt = gt_image == label
                        gt_list.append(gt)
                        gt_bg = np.logical_or(gt_bg, gt)

                    gt_image = np.dstack(
                        [np.invert(gt_bg), *gt_list]).astype(np.float32)

                    images.append(image)
                    gt_images.append(gt_image)

                yield np.array(images), np.array(gt_images)
        return get_batches_fn

    def gen_inference_batch_function(self, use_training_images=False):
        """
        Generate function to create batches of training data
        :return:
        """
        if use_training_images:
            image_paths = self.image_paths
        else:
            image_paths = self.test_image_paths
        image_shape = self.image_shape

        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                names = []
                shapes = []
                for image_file in image_paths[batch_i:batch_i + batch_size]:
                    print("Reading: ", image_file)
                    image = scipy.misc.imread(image_file, mode='RGB')
                    org_shape = image.shape
                    image = scipy.misc.imresize(image, image_shape)

                    images.append(image)
                    names.append(image_file)
                    shapes.append(org_shape)

                yield np.array(images), names, shapes
        return get_batches_fn
    
    def color_inference(self, image, labels, org_shape):
        
        image_shape = image.shape
        painted_image = np.zeros((image_shape[0], image_shape[1], 4))
        for i in range(self.get_num_classes()):
            # Paint at half transparency
            painted_image[labels == i] = (*self.gt_colors[i], 127)

        mask = scipy.misc.toimage(painted_image, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        street_im = scipy.misc.imresize(street_im, org_shape)
        
        return np.array(street_im)
                    
    def gen_test_output(self, sess, logits, keep_prob, image_pl):
        """
        Generate test output using the test images
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        files = self.test_image_paths
        image_shape = self.image_shape

        for i, file in enumerate(files):

            image = scipy.misc.imresize(
                scipy.misc.imread(file), image_shape)

            labels = sess.run(
                [logits],
                {keep_prob: 1.0, image_pl: [image]})

            labels = labels[0].reshape(image_shape[0], image_shape[1])
            painted_image = np.zeros((image_shape[0], image_shape[1], 4))
            for i in range(self.get_num_classes()):
                # Paint at half transparency
                painted_image[labels == i] = (*self.gt_colors[i], 127) #(*paints[paint], 127))

            mask = scipy.misc.toimage(painted_image, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            yield os.path.basename(file), np.array(street_im)

    def save_inference_samples(self, runs_dir, sess, logits, keep_prob, input_image):
        """
        Generate test output using the test images
        :param runs_dir: Folder to place output
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Run NN on test images and save them to HD
        print('Training Finished. Saving test images to: {}'.format(output_dir))
        image_outputs = self.gen_test_output(
            sess, logits, keep_prob, input_image)
        for name, image in image_outputs:
            scipy.misc.imsave(os.path.join(output_dir, name), image)
            
class LyftDataset(Dataset):

    '''
    class attributes
    '''
    images_path = "leftImg8bit"
    gt_path = "gtFine"
    gt_trail = "gtFine_color"
    image_trail = images_path
    training_list = []
    validation_list = []
    test_list = []

    def __init__(self, data_folder, image_shape):
        '''
        :param data_folder: Path to folder that contains all the datasets
        :param image_shape: Tuple - Shape of image
        '''
        self.data_folder = data_folder
        self.image_shape = image_shape
        self.gt_colors = [np.array((0, 0, 0, 0)), #gb 255
        				   np.array((0, 255, 0, 127)), # road 0
        				   np.array((255, 0, 0, 127))] # car
        self.gt_list = [255,0,13]

        self.image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        self.label_paths = {
            os.path.basename(path): path
            for path in glob(os.path.join(data_folder, 'Seg8bit', '*.png'))}

    def get_num_classes(self):
        return len(self.gt_list)

    def gen_batch_function(self):
        """
        Generate function to create batches of training data
        :return:
        """
        image_paths = self.image_paths
        label_paths = self.label_paths 
        image_shape = self.image_shape

        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in image_paths[batch_i:batch_i + batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(
                        scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(
                        scipy.misc.imread(gt_image_file), image_shape)

                    gt_bg = np.zeros(
                        [image_shape[0], image_shape[1]], dtype=bool)
                    gt_list = []
                    for label in self.gt_list[1:]: #auto bg
                        gt = gt_image == label
                        gt_list.append(gt)
                        gt_bg = np.logical_or(gt_bg, gt)

                    gt_image = np.dstack(
                        [np.invert(gt_bg), *gt_list]).astype(np.float32)

                    images.append(image)
                    gt_images.append(gt_image)

                yield np.array(images), np.array(gt_images)
        return get_batches_fn

    def gen_inference_batch_function(self):
        """
        Generate function to create batches of training data
        :return:
        """
        image_paths = self.image_paths
        image_shape = self.image_shape

        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                for image_file in image_paths[batch_i:batch_i + batch_size]:

                    image = scipy.misc.imresize(
                        scipy.misc.imread(image_file), image_shape)

                    images.append(image)

                yield np.array(images)
        return get_batches_fn
    
    def color_inference(self, image, labels):
        
        image_shape = image.shape
        painted_image = np.zeros((image_shape[0], image_shape[1], 4))
        for i in range(self.get_num_classes()):
            # Paint at half transparency
            painted_image[labels == i] = self.gt_colors[i]

        mask = scipy.misc.toimage(painted_image, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        
        return np.array(street_im)
                    
    def gen_test_output(self, sess, logits, keep_prob, image_pl):
        """
        Generate test output using the test images
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        files = self.image_paths
        image_shape = self.image_shape

        for file in files:
            image = scipy.misc.imresize(
                scipy.misc.imread(file), image_shape)

            labels = sess.run(
                [tf.argmax(tf.nn.softmax(logits), axis=-1)],
                {keep_prob: 1.0, image_pl: [image]})

            labels = labels[0].reshape(image_shape[0], image_shape[1])
            painted_image = np.zeros((image_shape[0], image_shape[1], 4))
            for i in range(self.get_num_classes()):
                # Paint at half transparency
                painted_image[labels == i] = self.gt_colors[i]

            mask = scipy.misc.toimage(painted_image, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            yield os.path.basename(file), np.array(street_im)

    def save_inference_samples(self, runs_dir, sess, logits, keep_prob, input_image):
        """
        Generate test output using the test images
        :param runs_dir: Folder to place output
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Run NN on test images and save them to HD
        print('Training Finished. Saving test images to: {}'.format(output_dir))
        image_outputs = self.gen_test_output(
            sess, logits, keep_prob, input_image)
        for name, image in image_outputs:
            scipy.misc.imsave(os.path.join(output_dir, name), image)

class kitti_2015(Dataset):

    '''
    class attributes
    '''
    num_classes = 2

    def __init__(self, data_folder, image_shape):
        self.data_folder = data_folder
        self.image_shape = image_shape

    def get_num_classes(self):
        return self.num_classes

    def gen_batch_function(self):
        """
        Generate function to create batches of training data
        :return:
        """
        data_folder = os.path.join(self.data_folder, 'datasets_kitti2015/training')
        image_shape = self.image_shape

        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
            label_paths = {
            os.path.basename(path): path
            for path in glob(os.path.join(data_folder, 'semantic', '*.png'))}
            
            road_color = 7
            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in image_paths[batch_i:batch_i + batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(
                        scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(
                        scipy.misc.imread(gt_image_file), image_shape)

                    gt_rd = gt_image == road_color
                    gt_bg = np.invert(gt_rd)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                    images.append(image)
                    gt_images.append(gt_image)

                yield np.array(images), np.array(gt_images)
        return get_batches_fn

    def gen_test_output(self, sess, logits, keep_prob, image_pl):
        """
        Generate test output using the test images
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        data_folder = os.path.join(self.data_folder, 'dataset_kitti2015/test')
        image_shape = self.image_shape

        for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
            image = scipy.misc.imresize(
                scipy.misc.imread(image_file), image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(
                image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(
                image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            yield os.path.basename(image_file), np.array(street_im)

    def save_inference_samples(self, runs_dir, sess, logits, keep_prob, input_image):
        """
        Generate test output using the test images
        :param runs_dir: Folder to place output
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Run NN on test images and save them to HD
        print('Training Finished. Saving test images to: {}'.format(output_dir))
        image_outputs = self.gen_test_output(
            sess, logits, keep_prob, input_image)
        for name, image in image_outputs:
            scipy.misc.imsave(os.path.join(output_dir, name), image)

class KittiDataset(Dataset):

    '''
    class attributes
    '''
    num_classes = 2

    def __init__(self, data_folder, image_shape):
        self.data_folder = data_folder
        self.image_shape = image_shape

    def get_num_classes(self):
        return self.num_classes

    def gen_batch_function(self):
        """
        Generate function to create batches of training data
        :return:
        """
        data_folder = os.path.join(self.data_folder, 'data_road/training')
        image_shape = self.image_shape

        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
            label_paths = {
                re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
            background_color = np.array([255, 0, 0])

            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in image_paths[batch_i:batch_i + batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(
                        scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(
                        scipy.misc.imread(gt_image_file), image_shape)

                    gt_bg = np.all(gt_image == background_color, axis=2)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    gt_image = np.concatenate(
                        (gt_bg, np.invert(gt_bg)), axis=2)

                    images.append(image)
                    gt_images.append(gt_image)

                yield np.array(images), np.array(gt_images)
        return get_batches_fn

    def gen_test_output(self, sess, logits, keep_prob, image_pl):
        """
        Generate test output using the test images
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        data_folder = os.path.join(self.data_folder, 'data_road/testing')
        image_shape = self.image_shape

        for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
            image = scipy.misc.imresize(
                scipy.misc.imread(image_file), image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(
                image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(
                image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            yield os.path.basename(image_file), np.array(street_im)

    def save_inference_samples(self, runs_dir, sess, logits, keep_prob, input_image):
        """
        Generate test output using the test images
        :param runs_dir: Folder to place output
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Run NN on test images and save them to HD
        print('Training Finished. Saving test images to: {}'.format(output_dir))
        image_outputs = self.gen_test_output(
            sess, logits, keep_prob, input_image)
        for name, image in image_outputs:
            scipy.misc.imsave(os.path.join(output_dir, name), image)


class CityscapeDataset(Dataset):

    '''
    class attributes
    '''
    images_path = "leftImg8bit"
    gt_path = "gtFine"
    gt_trail = "gtFine_color"
    image_trail = images_path
    training_list = []
    validation_list = []
    test_list = []

    def __init__(self, data_folder, image_shape):
        '''
        :param data_folder: Path to folder that contains all the datasets
        :param image_shape: Tuple - Shape of image
        '''
        self.data_folder = data_folder
        self.image_shape = image_shape
        self.gt_colors = {label: np.array(gt.color)
                          for label, gt in enumerate(cityscape_labels.labels)}

        combined_file_list = [[], [], []]
        sub_folders = ["train", "val", "test"]
        for i, folder in enumerate(sub_folders):
            image_full_path = os.path.join(data_folder,
                                           CityscapeDataset.images_path, folder)
            gt_full_path = os.path.join(
                data_folder, CityscapeDataset.gt_path, folder)
            # Recursively look for png files
            files = glob(os.path.join(image_full_path,
                                      '**', '*.png'), recursive=True)
            for file in files:
                relative_path = file[len(image_full_path):]
                tail = os.path.basename(relative_path)
                gt_base_folder = tail.replace(
                    CityscapeDataset.image_trail, CityscapeDataset.gt_trail)

                directory = os.path.dirname(relative_path)
                gt_expected_file = (
                    gt_full_path + '/' + directory + '/' + gt_base_folder)

                if os.path.exists(gt_expected_file):
                    combined_file_list[i].append((file, gt_expected_file))
        self.training_list = combined_file_list[0]
        self.validation_list = combined_file_list[1]
        self.test_list = combined_file_list[2]

    def get_num_classes(self):
        return len(cityscape_labels.labels)

    def gen_batch_function(self):
        """
        Generate function to create batches of training data
        :return:
        """
        image_paths = self.training_list
        image_shape = self.image_shape

        def get_batches_fn(batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                files = image_paths[batch_i:batch_i + batch_size]

                images = []
                gt_images = []

                for file in files:
                    image = scipy.misc.imresize(
                        scipy.misc.imread(file[0]), image_shape)
                    gt_image = scipy.misc.imresize(
                        scipy.misc.imread(file[1], mode='RGB'), image_shape)

                    gt_bg = np.zeros(
                        [image_shape[0], image_shape[1]], dtype=bool)
                    gt_list = []
                    for label in cityscape_labels.labels[1:]:
                        gt = np.all(
                            gt_image == np.array(label.color), axis=2)
                        gt_list.append(gt)
                        gt_bg = np.logical_or(gt_bg, gt)

                    gt_image = np.dstack(
                        [np.invert(gt_bg), *gt_list]).astype(np.float32)

                    images.append(image)
                    gt_images.append(gt_image)

                yield np.array(images), np.array(gt_images)
        return get_batches_fn

    def gen_test_output(self, sess, logits, keep_prob, image_pl):
        """
        Generate test output using the test images
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        files = self.test_list
        image_shape = self.image_shape
        paints = self.gt_colors

        for file in files:
            image = scipy.misc.imresize(
                scipy.misc.imread(file[0]), image_shape)
            gt_image = scipy.misc.imresize(
                scipy.misc.imread(file[1]), image_shape)

            labels = sess.run(
                [tf.argmax(tf.nn.softmax(logits), axis=-1)],
                {keep_prob: 1.0, image_pl: [image]})

            labels = labels[0].reshape(image_shape[0], image_shape[1])
            painted_image = np.zeros_like(gt_image)
            for paint in paints:
                # Paint at half transparency
                painted_image[labels == paint] = np.array(
                    (*paints[paint], 127))

            mask = scipy.misc.toimage(painted_image, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            yield os.path.basename(file[0]), np.array(street_im)

    def save_inference_samples(self, runs_dir, sess, logits, keep_prob, input_image):
        """
        Generate test output using the test images
        :param runs_dir: Folder to place output
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep robability
        :param image_pl: TF Placeholder for the image placeholder
        :return: Output for for each test image
        """
        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Run NN on test images and save them to HD
        print('Training Finished. Saving test images to: {}'.format(output_dir))
        image_outputs = self.gen_test_output(
            sess, logits, keep_prob, input_image)
        for name, image in image_outputs:
            scipy.misc.imsave(os.path.join(output_dir, name), image)


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [
        vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))
