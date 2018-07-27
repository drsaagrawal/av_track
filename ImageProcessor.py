'''
Created on March 18, 2018

@author: Asad Zia
'''

import numpy as np
import scipy.misc
import tensorflow as tf


class ImageProcessor(object):

    '''
    class attributes
    '''

    def __init__(self, image_shape, sess, logits, keep_prob, input_image):
        '''
        Constructor
        '''
        self.sess = sess
        self.logits = logits
        self.keep_prob = keep_prob
        self.input_image = input_image
        self.image_shape = image_shape

    # Video processing pipeline
    def process_image(self, image):
        '''
        Kitti support only
        '''
        sess = self.sess
        logits = self.logits
        keep_prob = self.keep_prob
        image_pl = self.input_image
        requied_shape = self.image_shape
        original_shape = image.shape

        image = scipy.misc.imresize(image, requied_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(
            requied_shape[0], requied_shape[1])
        segmentation = (im_softmax > 0.5).reshape(
            requied_shape[0], requied_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        # Restore orginal image shape
        return scipy.misc.imresize(np.array(street_im), original_shape)
