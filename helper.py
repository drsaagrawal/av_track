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
        self.gt_colors = {7:np.array((128, 64,128))}
		# {label: np.array(gt.color)
         #                  for label, gt in enumerate(cityscape_labels.labels)}

        combined_file_list = [[], [], []]
        print(self.gt_colors)
        sub_folders = ["train", "val", "test"]
        pune_folder = "pune"
        for i, folder in enumerate(sub_folders):
            if folder=="test":
                image_full_path = os.path.join(data_folder,
                                               CityscapeDataset.images_path, folder,pune_folder)
            else:
                image_full_path = os.path.join(data_folder,
                                               CityscapeDataset.images_path, folder)
            print("Image Path",image_full_path)
            gt_full_path = os.path.join(
                data_folder, CityscapeDataset.gt_path, folder)
            # Recursively look for png files
            if folder == "test":
                # print("Reading pune images")
                files = glob(os.path.join(image_full_path,
                                          '**', '*.jpg'), recursive=True)
                print("Total number of pune images:",len(files))
            else:
                files = glob(os.path.join(image_full_path,
                                      '**', '*.png'), recursive=True)
            for file in files:
                relative_path = file[len(image_full_path):]
                # print("relative path",relative_path)
                tail = os.path.basename(relative_path)
                # print("Tail :",tail)
                gt_base_folder = tail.replace(
                    CityscapeDataset.image_trail, CityscapeDataset.gt_trail)
                # print("gt_base_folder :",gt_base_folder)

                directory = os.path.dirname(relative_path)
                # print("directory",directory)
                gt_expected_file = (
                    gt_full_path + '/' + directory + '/' + gt_base_folder)
                # print("gt_expected_file :",gt_expected_file)
                if os.path.exists(gt_expected_file):
                    combined_file_list[i].append((file, gt_expected_file))
                else:
                    combined_file_list[i].append((file, ''))
        self.training_list = combined_file_list[0]
        self.validation_list = combined_file_list[1]
        self.test_list = combined_file_list[2]
        print("Test dataset",self.test_list[1])

    def get_num_classes(self):
        return 2
        # return len(cityscape_labels.labels)

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
                        if label.name == "road":
                            gt_bg = np.all(gt_image == np.array(label.color), axis=2)
                            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                            gt_image = np.concatenate(
                                (gt_bg, np.invert(gt_bg)), axis=2)
                            # gt = np.all(
                            # gt_image == np.array(label.color), axis=2)
                            # gt_list.append(gt)
                            # gt_bg = np.logical_or(gt_bg, gt)


                    # gt_image = np.dstack(
                    #     [np.invert(gt_bg), *gt_list]).astype(np.float32)
                    # print("gt_image",gt_image.shape)

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
            if file[1] != '':
                gt_image = scipy.misc.imresize(
                    scipy.misc.imread(file[1]), image_shape)

            #labels = sess.run(
                #[tf.argmax(tf.nn.softmax(logits), axis=-1)],
                #{keep_prob: 1.0, image_pl: [image]})
            # print("labels",len(labels[0]),"la",labels[0])
            #print(len(labels))
            #print("labels len", len(labels))
            #print("labels shape", labels.size())
            #labels = labels[0].reshape(image_shape[0], image_shape[1])
            #painted_image = np.zeros_like(gt_image)
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(
                image_shape[0], image_shape[1])
            segmentation = (im_softmax <= 0.5).reshape(
                image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            yield os.path.basename(file[0]), np.array(street_im)
            # for paint in paints:
            #     # Paint at half transparency
            #     painted_image[labels == paint] = np.array(
            #         (*paints[paint], 127))
			#
            # mask = scipy.misc.toimage(painted_image, mode="RGBA")
            # street_im = scipy.misc.toimage(image)
            # street_im.paste(mask, box=None, mask=mask)
			#
            # yield os.path.basename(file[0]), np.array(street_im)




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