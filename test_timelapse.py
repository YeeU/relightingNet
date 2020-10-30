import os
import numpy as np
import tensorflow as tf
import shutil
from PIL import Image
from skimage import io
from collections import namedtuple
import argparse

from modules import renderingNet, sdNet, lambSH_layer, irn_layer, spade_models


img1_path = './timeLapse_imgs/1.png'
img2_path = './timeLapse_imgs/2.png'
mask_path = './timeLapse_imgs/mask.png'

img1 = io.imread(img1_path)
img2 = io.imread(img2_path)
mask = io.imread(mask_path); mask = mask if mask.ndim == 3 else np.tile(mask[...,None], (1,1,3))

dst_dir = './timeLapse_rendering.png'

rendering_model_path = 'relight_model/model.ckpt'
skyGen_model_path = 'model_skyGen_net/model.ckpt'



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', '0.001', """learning rate""")
tf.app.flags.DEFINE_float('beta1', '0.5', """beta for Adam""")
tf.app.flags.DEFINE_integer('batch_size', '5', """batch size""")
tf.app.flags.DEFINE_integer('c_dim', '3', """c dimsion""")
tf.app.flags.DEFINE_integer('z_dim', '64', """z dimsion""")
tf.app.flags.DEFINE_integer('output_size', '200', """output size""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")

 
def new_size(img):
	img_h, img_w = img.shape[:2]
	if img_h > img_w:
		scale = img_w / 200
		new_img_h = np.int32(img_h / scale)
		new_img_w = 200
	else:
		scale = img_h / 200
		new_img_w = np.int32(img_w / scale)
		new_img_h = 200

	return new_img_h, new_img_w

def resize_img(img, new_img_h, new_img_w):
	scale = img.shape[0] / new_img_h

	u, v = np.meshgrid(np.arange(new_img_w), np.arange(new_img_h))
	x = np.int32(u * scale)
	y = np.int32(v * scale)
	
	new_img = np.zeros((new_img_h, new_img_w, 3), np.float32)
	new_img[v,u] = img[y,x]

	return new_img


def init_sd(nm_irn, lightings):
	flatten_lightings = tf.tile(tf.reshape(lightings, (-1,1,1,27)), (1,tf.shape(nm_irn)[1],tf.shape(nm_irn)[2],1))

	return tf.concat([nm_irn, flatten_lightings], axis=-1)


def build_model(input_height, input_width):

	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
		input1_var = tf.placeholder(tf.float32, (None, input_height, input_width, 3))
		input2_var = tf.placeholder(tf.float32, (None, input_height, input_width, 3))
		mask_var = tf.placeholder(tf.float32, (None, input_height, input_width, 1))
		train_flag = tf.placeholder(tf.bool, ())

		input1_noSky = input1_var * mask_var
		input2_noSky = input2_var * mask_var


		input_shape = [5, input_height, input_width, 3]
		irnLayer = irn_layer.Irn_layer(input_shape, train_flag)

		_, _, _, new_lighting = irnLayer(input1_noSky, mask_var)
		albedo, shadow, nm_pred, lighting = irnLayer(input2_noSky, mask_var)


		shading, rendering_mask = lambSH_layer.lambSH_layer(tf.ones_like(albedo), nm_pred, new_lighting, tf.ones_like(shadow), 1.)

		rendering = tf.pow(albedo * shading * shadow, 1/2.2) * mask_var
		residual = input2_noSky - rendering


		OPTIONS = namedtuple('OPTIONS', ['output_c_dim', 'is_training', 'gf_dim'])
		options = OPTIONS(output_c_dim = 1, is_training = True, gf_dim = 8)
		shadow_gen = sdNet.shadow_generator(init_sd(nm_pred, new_lighting), options, name='sd_generator')


		g_input = tf.concat([albedo, nm_pred, shading, residual, shadow_gen, 1-mask_var], axis=-1)

		relit_rendering = renderingNet.rendering_Net(inputs=g_input, masks=mask_var, is_training=train_flag, height=input_height, width=input_width, n_layers=30, n_pools=4, depth_base=32)

		## generate background sky image ##
		init_sky = tf.random_normal((tf.shape(relit_rendering)[0], FLAGS.z_dim*4), dtype=tf.float32)
		cinput_sky1 = tf.image.resize_images((relit_rendering*2.-1.)*mask_var, (200,200))
		cinput_sky2 = tf.image.resize_images(1.-mask_var, (200,200))
		cinput_sky = tf.concat([cinput_sky1, cinput_sky2], axis=-1)
		sky = spade_models.generator(init_sky, cinput_sky, train_flag)
		sky = tf.image.resize_images(sky, (input_height, input_width))

		rendering = relit_rendering * mask_var + sky * (1.-mask_var)

		return rendering, input1_var, input2_var, mask_var, train_flag


new_img_h, new_img_w = new_size(img1)
img1 = resize_img(img1/255., new_img_h, new_img_w)
img2 = resize_img(img2/255., new_img_h, new_img_w)
mask = resize_img(mask/255., new_img_h, new_img_w)
img1 = img1[None]
img2 = img2[None]
img_mask = mask[None,...,:1]

rendering, input1_var, input2_var, mask_var, train_flag = build_model(new_img_h, new_img_w)


irn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='inverserendernet')
rn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
sd_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sd_generator')
sg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sky_generator')
rendering_vars = rn_vars + irn_vars + sd_vars


sess = tf.InteractiveSession()
rendering_saver = tf.train.Saver(rendering_vars)
rendering_saver.restore(sess, rendering_model_path)
skyGen_saver = tf.train.Saver(sg_vars)
skyGen_saver.restore(sess, skyGen_model_path)


[rendering_val] = sess.run([rendering], feed_dict={train_flag:False, input1_var:img1, input2_var:img2, mask_var:img_mask})

rendering_val = np.uint8(rendering_val[0]*255.)
io.imsave(dst_dir, rendering_val)




