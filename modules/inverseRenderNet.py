from __future__ import division
import tensorflow as tf
from modules.ops import *
from modules.utils import *


def irn_resnet(image, options, name="inverserendernet"):
    with tf.variable_scope(name):

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x


        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))

        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')


        am1 = tf.image.resize_bilinear(r9, c2.shape[1:3])
        am1 = conv2d(am1, options.gf_dim*2, 3, 1, name='g_am1_dc')
        am1 = tf.nn.relu(instance_norm(am1, 'g_am1_bn'))
        
        am2 = tf.image.resize_bilinear(am1, c1.shape[1:3])
        am2 = conv2d(am2, options.gf_dim, 3, 1, name='g_am2_dc')
        am2 = tf.nn.relu(instance_norm(am2, 'g_am2_bn'))
        am2 = tf.pad(am2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        
        am_out = conv2d(am2, options.am_out_c_dim, 7, 1, padding='VALID', name='g_am_out_c')


        nm1 = tf.image.resize_bilinear(r9, c2.shape[1:3])
        nm1 = conv2d(nm1, options.gf_dim*2, 3, 1, name='g_nm1_dc')
        nm1 = tf.nn.relu(instance_norm(nm1, 'g_nm1_bn'))
        
        nm2 = tf.image.resize_bilinear(nm1, c1.shape[1:3])
        nm2 = conv2d(nm2, options.gf_dim, 3, 1, name='g_nm2_dc')
        nm2 = tf.nn.relu(instance_norm(nm2, 'g_nm2_bn'))
        nm2 = tf.pad(nm2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        
        nm_out = conv2d(nm2, options.nm_out_c_dim, 7, 1, padding='VALID', name='g_nm_out_c')


        mask1 = tf.image.resize_bilinear(r9, c2.shape[1:3])
        mask1 = conv2d(mask1, options.gf_dim*2, 3, 1, name='g_mask1_dc')
        mask1 = tf.nn.relu(instance_norm(mask1, 'g_mask1_bn'))

        mask2 = tf.image.resize_bilinear(mask1, c1.shape[1:3])
        mask2 = conv2d(mask2, options.gf_dim, 3, 1, name='g_mask2_dc')
        mask2 = tf.nn.relu(instance_norm(mask2, 'g_mask2_bn'))
        mask2 = tf.pad(mask2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        
        mask_out = conv2d(mask2, options.mask_out_c_dim, 7, 1, padding='VALID', name='g_mask_out_c')

        return am_out, mask_out, nm_out


