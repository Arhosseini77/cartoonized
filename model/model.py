# model/model.py
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def resblock(inputs, out_channel=32, name='resblock'):
    with tf.variable_scope(name):
        x = slim.convolution2d(inputs, out_channel, [3, 3],
                               activation_fn=None, scope='conv1')
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3],
                               activation_fn=None, scope='conv2')
        return x + inputs


def unet_generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x0 = tf.nn.leaky_relu(x0)

        x1 = slim.convolution2d(x0, channel, [3, 3], stride=2, activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)
        x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)

        x2 = slim.convolution2d(x1, channel * 2, [3, 3], stride=2, activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)
        x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        for idx in range(num_blocks):
            x2 = resblock(x2, out_channel=channel * 4, name='block_{}'.format(idx))

        x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
        x3 = slim.convolution2d(x3 + x1, channel * 2, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)
        x3 = slim.convolution2d(x3, channel, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
        x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
        x4 = tf.nn.leaky_relu(x4)
        x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)
        return x4


def tf_box_filter(x, r):
    k_size = int(2 * r + 1)
    ch = x.get_shape().as_list()[-1]
    weight = 1 / (k_size ** 2)
    box_kernel = weight * np.ones((k_size, k_size, ch, 1), dtype=np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, strides=[1, 1, 1, 1], padding='SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)
    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)
    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x
    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x
    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N
    output = mean_A * x + mean_b
    return output


class CartoonizerModel:
    def __init__(self, model_path, channel=32, num_blocks=4):
        self.model_path = model_path
        self.channel = channel
        self.num_blocks = num_blocks
        self.sess = None
        self.input_photo = None
        self.output = None
        self._build_model()

    def _build_model(self):
        self.input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_photo')
        network_out = unet_generator(self.input_photo, channel=self.channel, num_blocks=self.num_blocks)
        self.output = guided_filter(self.input_photo, network_out, r=1, eps=5e-3)

    def load_model(self):
        saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if 'generator' in var.name])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        # Load weights from the "weights" folder
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))

    def cartoonize(self, image):
        # image should be a numpy array (HxWx3) with values in [-1,1]
        batch_image = np.expand_dims(image, axis=0)
        output = self.sess.run(self.output, feed_dict={self.input_photo: batch_image})
        output = np.squeeze(output)
        return output
