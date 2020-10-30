import tensorflow as tf
import importlib
import numpy as np
import os
from collections import namedtuple

from modules import pred_illuDecomp_layer, inverseRenderNet


class Irn_layer(object):
    def __init__(self, inputs_shape, train_flag):
        super(Irn_layer, self).__init__()
        self.inverseRenderNet = inverseRenderNet.irn_resnet
        
        self.height = inputs_shape[1]
        self.width = inputs_shape[2]
        self.train_flag = train_flag

        OPTIONS = namedtuple('OPTIONS', ['am_out_c_dim', 'nm_out_c_dim', 'mask_out_c_dim', 'is_training', 'gf_dim'])
        self.options = OPTIONS(am_out_c_dim=3, nm_out_c_dim=2, mask_out_c_dim=1, is_training=False, gf_dim = 64)

    def __call__(self, inputs_var, masks_var):

        am_deconv_out, mask_deconv_out, nm_deconv_out = self.inverseRenderNet(inputs_var, self.options)


        # post-process am, shadow and nm
        albedos = tf.nn.sigmoid(am_deconv_out) * masks_var + tf.constant(1e-4)
        shadow = tf.nn.sigmoid(mask_deconv_out) * masks_var + tf.constant(1e-4)


        nm_pred = nm_deconv_out
        nm_pred_norm = tf.sqrt(tf.reduce_sum(nm_pred**2, axis=-1, keepdims=True)+tf.constant(1.))
        nm_pred_xy = nm_pred / nm_pred_norm
        nm_pred_z = tf.constant(1.) / nm_pred_norm
        nm_pred = tf.concat([nm_pred_xy, nm_pred_z], axis=-1) * masks_var

        # infer lightings
        gamma = tf.constant(2.2)

        lighting_model = 'illu_pca'
        lighting_vectors = tf.constant(np.load(os.path.join(lighting_model,'pcaVector.npy')),dtype=tf.float32)
        lighting_means = tf.constant(np.load(os.path.join(lighting_model,'mean.npy')),dtype=tf.float32)
        lightings_var = tf.constant(np.load(os.path.join(lighting_model,'pcaVariance.npy')),dtype=tf.float32)
        

        lightings = pred_illuDecomp_layer.illuDecomp(inputs_var,albedos,nm_pred,shadow,gamma,masks_var)
        lightings_pca = tf.matmul((lightings - lighting_means), pinv(lighting_vectors))

        # recompute lightings from lightins_pca which could add weak constraint on lighting reconstruction 
        lightings = tf.matmul(lightings_pca,lighting_vectors) + lighting_means 

        # reshape 27-D lightings to 9*3 lightings
        lightings = tf.reshape(lightings,[tf.shape(lightings)[0],9,3])


        return albedos, shadow, nm_pred, lightings
    


# compute pseudo inverse for input matrix
def pinv(A, reltol=1e-6):
	# compute SVD of input A
	s, u, v = tf.svd(A)

	# invert s and clear entries lower than reltol*s_max
	atol = tf.reduce_max(s) * reltol
	s = tf.where(s>atol, s, atol*tf.ones_like(s))
	s_inv = tf.diag(1./s)

	# compute v * s_inv * u_t as psuedo inverse
	return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))



