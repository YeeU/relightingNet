from __future__ import division
import tensorflow as tf
from modules.ops import *
from modules.utils import *


def shadow_generator(inputs, options, name="generator"):

	with tf.variable_scope(name):

		e1 = tf.nn.relu(conv2d(inputs, options.gf_dim*8, 3, 1, name='g_e1_conv'))
		e2 = tf.nn.max_pool(batch_norm(e1, options.is_training, 'g_bn_e1'), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		e2 = tf.nn.relu(conv2d(e2, options.gf_dim*16, 3, 1, name='g_e2_conv'))
		e3 = tf.nn.max_pool(batch_norm(e2, options.is_training, 'g_bn_e2'), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		e3 = tf.nn.relu(conv2d(e3, options.gf_dim*32, 3, 1, name='g_e3_conv'))
		e4 = tf.nn.max_pool(batch_norm(e3, options.is_training, 'g_bn_e3'), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		e4 = tf.nn.relu(conv2d(e4, options.gf_dim*64, 3, 1, name='g_e4_conv'))
		e5 = tf.nn.max_pool(batch_norm(e4, options.is_training, 'g_bn_e4'), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		e5 = tf.nn.relu(conv2d(e5, options.gf_dim*64, 3, 1, name='g_e5_conv'))
		e6 = tf.nn.max_pool(batch_norm(e5, options.is_training, 'g_bn_e5'), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		e6 = tf.nn.relu(conv2d(e6, options.gf_dim*64, 3, 1, name='g_e6_conv'))
		e7 = tf.nn.max_pool(batch_norm(e6, options.is_training, 'g_bn_e6'), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		e7 = tf.nn.relu(conv2d(e7, options.gf_dim*64, 3, 1, name='g_e7_conv'))
		e8 = tf.nn.max_pool(batch_norm(e7, options.is_training, 'g_bn_e7'), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		e8 = tf.nn.relu(conv2d(e8, options.gf_dim*64, 3, 1, name='g_e8_conv'))

		d1 = tf.image.resize_images(e8, e7.shape[1:3])
		d1 = conv2d(batch_norm(d1, options.is_training, 'g_'+name+'_bn_d1'), options.gf_dim*64, 3, 1, name='g_'+name+'d1_dc')
		d1 = tf.concat([tf.nn.relu(d1), e7], 3)

		d2 = tf.image.resize_images(d1, e6.shape[1:3])
		d2 = conv2d(batch_norm(d2, options.is_training, 'g_'+name+'_bn_d2'), options.gf_dim*64, 3, 1, name='g_'+name+'d2_dc')
		d2 = tf.concat([tf.nn.relu(d2), e6], 3)

		d3 = tf.image.resize_images(d2, e5.shape[1:3])
		d3 = conv2d(batch_norm(d3, options.is_training, 'g_'+name+'_bn_d3'), options.gf_dim*64, 3, 1, name='g_'+name+'d3_dc')
		d3 = tf.concat([tf.nn.relu(d3), e5], 3)

		d4 = tf.image.resize_images(d3, e4.shape[1:3])
		d4 = conv2d(batch_norm(d4, options.is_training, 'g_'+name+'_bn_d4'), options.gf_dim*64, 3, 1, name='g_'+name+'d4_dc')
		d4 = tf.concat([tf.nn.relu(d4), e4], 3)

		d5 = tf.image.resize_images(d4, e3.shape[1:3])
		d5 = conv2d(batch_norm(d5, options.is_training, 'g_'+name+'_bn_d5'), options.gf_dim*32, 3, 1, name='g_'+name+'d5_dc')
		d5 = tf.concat([tf.nn.relu(d5), e3], 3)

		d6 = tf.image.resize_images(d5, e2.shape[1:3])
		d6 = conv2d(batch_norm(d6, options.is_training, 'g_'+name+'_bn_d6'), options.gf_dim*16, 3, 1, name='g_'+name+'d6_dc')
		d6 = tf.concat([tf.nn.relu(d6), e2], 3)

		d7 = tf.image.resize_images(d6, e1.shape[1:3])
		d7 = conv2d(batch_norm(d7, options.is_training, 'g_'+name+'_bn_d7'), options.gf_dim*8, 3, 1, name='g_'+name+'d7_dc')
		d7 = tf.concat([tf.nn.relu(d7), e1], 3)

		d8 = tf.image.resize_images(d7, inputs.shape[1:3])
		d8 = conv2d(batch_norm(d8, options.is_training, 'g_'+name+'_bn_d8'), options.output_c_dim, 3, 1, name='g_'+name+'d8_dc')

		pred = tf.clip_by_value(tf.nn.tanh(d8), -.9999, .9999)
		pred = pred / 2. + .5


		return pred

