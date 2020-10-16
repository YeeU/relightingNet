import importlib
import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.layers as layers


def rendering_Net(inputs, masks, height, width, n_layers=12, n_pools=2, is_training=True, depth_base=64):
	conv_layers = np.int32(n_layers/2) -1
	deconv_layers = np.int32(n_layers/2)
	# number of layers before perform pooling
	nlayers_befPool = np.int32(np.ceil((conv_layers-1)/n_pools)-1)

	max_depth = 512

	if depth_base*2**n_pools < max_depth:
		tail = conv_layers - nlayers_befPool*n_pools


		tail_deconv = deconv_layers - nlayers_befPool*n_pools
	else:
		maxNum_pool = np.log2(max_depth / depth_base)
		tail = np.int32(conv_layers - nlayers_befPool * maxNum_pool)
		tail_deconv = np.int32(deconv_layers - nlayers_befPool * maxNum_pool)

	f_in_conv = [3] + [np.int32(depth_base*2**(np.ceil(i/nlayers_befPool)-1)) for i in range(1, conv_layers-tail+1)] + [np.int32(depth_base*2**maxNum_pool) for i in range(conv_layers-tail+1, conv_layers+1)]
	f_out_conv = [64] + [np.int32(depth_base*2**(np.floor(i/nlayers_befPool))) for i in range(1, conv_layers-tail+1)] + [np.int32(depth_base*2**maxNum_pool) for i in range(conv_layers-tail+1, conv_layers+1)]

	f_in_deconv = f_out_conv[:0:-1] + [64]
	f_out_amDeconv = f_in_conv[:0:-1] + [3]
	f_out_MaskDeconv = f_in_conv[:0:-1] + [1]
	f_out_nmDeconv = f_in_conv[:0:-1] + [2]


	### contractive conv_layer block
	conv_out = inputs
	conv_out_list = []
	for i,f_in,f_out in zip(range(1,conv_layers+2),f_in_conv,f_out_conv):
		scope = 'generator/conv'+str(i)

		if np.mod(i-1,nlayers_befPool)==0 and i<=n_pools*nlayers_befPool+1 and i != 1:
			conv_out_list.append(conv_out)
			conv_out = conv2d(conv_out, scope, f_in, f_out)
			conv_out = tf.nn.max_pool(conv_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		else:
			conv_out = conv2d(conv_out, scope, f_in, f_out)


	### expanding deconv_layer block succeeding conv_layer block
	deconv_out = conv_out
	for i,f_in,f_out in zip(range(1,deconv_layers+1),f_in_deconv,f_out_amDeconv):
		scope = 'generator/deconv'+str(i)

		# expand resolution every after nlayers_befPool deconv_layer
		if np.mod(i,nlayers_befPool)==0 and i<=n_pools*nlayers_befPool:
			tmp = conv_out_list[-np.int32(i/nlayers_befPool)]
			deconv_out = conv2d(tf.image.resize_bilinear(deconv_out, tmp.shape[1:3]), scope, f_in, f_out)
			tmp = conv2d(tmp, scope+'/concat', f_in, f_out)
			deconv_out = tmp + deconv_out


		elif i==deconv_layers:
			deconv_out = layers.conv2d(deconv_out,num_outputs=f_out,kernel_size=[3,3],stride=[1,1],padding='SAME',normalizer_fn=None,activation_fn=None,weights_initializer=tf.random_normal_initializer(mean=0,stddev=np.sqrt(2/9/f_in)),weights_regularizer=layers.l2_regularizer(scale=1e-5),scope=scope)


		else:
			# layers that not expand spatial resolution
			deconv_out = conv2d(deconv_out, scope, f_in, f_out)


	return tf.clip_by_value(tf.nn.sigmoid(deconv_out), 1e-4, .9999)
 

def conv2d(inputs, scope, f_in, f_out):
	conv_out = layers.conv2d(inputs, num_outputs=f_out, kernel_size=[3,3],stride=[1,1],padding='SAME',normalizer_fn=None, activation_fn=None, weights_initializer=tf.random_normal_initializer(mean=0,stddev=np.sqrt(2/9/f_in)), weights_regularizer=layers.l2_regularizer(scale=1e-5),biases_initializer=None, scope=scope)

	with tf.variable_scope(scope):
		gn_out = group_norm(conv_out)

		relu_out = tf.nn.relu(gn_out)

	return relu_out


def group_norm(inputs, scope='group_norm'):
	input_shape = tf.shape(inputs)
	_,H,W,C = inputs.get_shape().as_list()
	group = 32
	with tf.variable_scope(scope):
		gamma = tf.get_variable('scale', shape=[C], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=True, regularizer=layers.l2_regularizer(scale=1e-5))

		beta = tf.get_variable('bias', shape=[C], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True, regularizer=layers.l2_regularizer(scale=1e-5))
	
		inputs = tf.reshape(inputs, [-1, H, W, group, C // group], name='unpack')
		mean, var = tf.nn.moments(inputs, [1, 2, 4], keep_dims=True)
		inputs = (inputs - mean) / tf.sqrt(var + 1e-5)
		inputs = tf.reshape(inputs, input_shape, name='pack')
		gamma = tf.reshape(gamma, [1, 1, 1, C], name='reshape_gamma')
		beta = tf.reshape(beta, [1, 1, 1, C], name='reshape_beta')
		return inputs * gamma + beta

