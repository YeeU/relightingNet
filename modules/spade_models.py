from modules.spade_ops import *
from modules.spade_utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np

FLAGS = tf.app.flags.FLAGS

def generator(noise, cInputs, random_style=False, reuse=False, scope="sky_generator"):
    # import ipdb; ipdb.set_trace()
    channel = FLAGS.z_dim * 4 * 4

    with tf.variable_scope(scope):
        x = fully_connected(noise, units=2 * 2 * channel, use_bias=True, sn=False, scope='linear_x')
        x = tf.reshape(x, [-1, 2, 2, channel])


        x = spade_resblock(cInputs, x, channels=channel, use_bias=True, sn=True, scope='spade_resblock_fix_0')

        shapes = [4,7,13,25,50,100,200]
        for i, shape in enumerate(shapes):
            x = up_sample(x, shape, shape)
            if i > 2:
                x = spade_resblock(cInputs, x, channels=channel//2, use_bias=True, sn=True, scope='spade_resblock_' + str(i))

                channel = channel // 2
                # 512 -> 256 -> 128 -> 64
            else:
                x = spade_resblock(cInputs, x, channels=channel, use_bias=True, sn=True, scope='spade_resblock_' + str(i))

        x = lrelu(x, 0.2)
        x = conv(x, channels=FLAGS.c_dim, kernel=3, stride=1, pad=1, use_bias=True, sn=False, scope='logit')
        x = tanh(x)

        x = x / 2. + .5

        return x

##################################################################################
# Discriminator
##################################################################################

def discriminator(segmap, x_init, reuse=False, scope="discriminator"):
    D_logit = []

    with tf.variable_scope(scope):
        for scale in range(2):
            feature_loss = []
            channel = FLAGS.z_dim
            x = tf.concat([segmap, x_init], axis=-1)

            x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, sn=False, scope='ms_' + str(scale) + 'conv_0')
            x = lrelu(x, 0.2)

            feature_loss.append(x)

            for i in range(1, 4):
                stride = 1 if i == 4 - 1 else 2

                x = conv(x, channel * 2, kernel=4, stride=stride, pad=1, use_bias=True, sn=True, scope='ms_' + str(scale) + 'conv_' + str(i))
                x = instance_norm(x, scope='ms_' + str(scale) + 'ins_norm_' + str(i))
                x = lrelu(x, 0.2)

                feature_loss.append(x)

                channel = min(channel * 2, 512)


            x = conv(x, channels=1, kernel=4, stride=1, pad=1, use_bias=True, sn=True, scope='ms_' + str(scale) + 'D_logit')

            feature_loss.append(x)
            D_logit.append(feature_loss)

            x_init = down_sample_avg(x_init)
            segmap = down_sample_avg(segmap)

        return D_logit

##################################################################################
# Model
##################################################################################

def image_translate(self, segmap_img, x_img=None, random_style=False, reuse=False):

    if random_style :
        x_mean, x_var = None, None
    else :
        x_mean, x_var = self.image_encoder(x_img, reuse=reuse, scope='encoder')

    x = self.generator(segmap_img, x_mean, x_var, random_style, reuse=reuse, scope='generator')

    return x, x_mean, x_var

def image_discriminate(self, segmap_img, real_img, fake_img):
    real_logit = self.discriminator(segmap_img, real_img, scope='discriminator')
    fake_logit = self.discriminator(segmap_img, fake_img, reuse=True, scope='discriminator')

    return real_logit, fake_logit

def gradient_penalty(self, real, segmap, fake):
    if self.gan_type == 'dragan':
        shape = tf.shape(real)
        eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
        x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
        noise = 0.5 * x_std * eps  # delta in paper

        alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
        interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

    else:
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = alpha * real + (1. - alpha) * fake

    logit = self.discriminator(segmap, interpolated, reuse=True, scope='discriminator')

    GP = []


    for i in range(self.n_scale) :
        grad = tf.gradients(logit[i][-1], interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

        # WGAN - LP
        if self.gan_type == 'wgan-lp':
            GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

    return tf.reduce_mean(GP)
