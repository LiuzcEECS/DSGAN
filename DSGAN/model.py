from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class CoGAN(object):
  def __init__(self, sess, L1_lambda=100, batch_size=1, sample_num=1, 
         input_width = 640, input_height = 480, crop_width=256, crop_height=256, input_c_dim=3,
         output_height=256, output_width=256, output_c_dim=2,
         gf_dim=64, df_dim=64, dataset_name='default',
         checkpoint_dir=None, sample_dir=None, is_crop=True):

    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [128]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [32]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [32]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """

    self.sess = sess
    self.L1_lambda = L1_lambda
    self.batch_size = batch_size
    self.sample_num = sample_num
    self.input_width = input_width
    self.input_height = input_height
    self.crop_width = crop_width
    self.crop_height = crop_height
    self.input_c_dim = input_c_dim
    self.output_width = output_width
    self.output_height = output_height
    self.output_c_dim = output_c_dim
    self.gf_dim = gf_dim
    self.df_dim = df_dim
    self.sample_dir = sample_dir
    self.checkpoint_dir = checkpoint_dir

    # ------------batch norm-------------------
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn4 = batch_norm(name='d_bn4')
    self.d_bn5 = batch_norm(name='d_bn5')

    self.g_bn_e2 = batch_norm(name='g_bn_e2')
    self.g_bn_e3 = batch_norm(name='g_bn_e3')
    self.g_bn_e4 = batch_norm(name='g_bn_e4')
    self.g_bn_e5 = batch_norm(name='g_bn_e5')
    self.g_bn_e6 = batch_norm(name='g_bn_e6')
    self.g_bn_e7 = batch_norm(name='g_bn_e7')
    self.g_bn_e8 = batch_norm(name='g_bn_e8')

    self.g_bn_d1 = batch_norm(name='g_bn_d1')
    self.g_bn_d2 = batch_norm(name='g_bn_d2')
    self.g_bn_d3 = batch_norm(name='g_bn_d3')
    self.g_bn_d4 = batch_norm(name='g_bn_d4')
    self.g_bn_d5 = batch_norm(name='g_bn_d5')
    self.g_bn_d6 = batch_norm(name='g_bn_d6')
    self.g_bn_d7 = batch_norm(name='g_bn_d7')
    self.g_bn_d8_1 = batch_norm(name='g_bn_d8_1')
    self.g_bn_d8_2 = batch_norm(name='g_bn_d8_2')
    self.g_bn_d9_1 = batch_norm(name='g_bn_d9_1')
    self.g_bn_d9_2 = batch_norm(name='g_bn_d9_2')
    # -----------------------------------------

    

    self.dataset_name = dataset_name
    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    
    image_rgb_dims = [self.crop_height, self.crop_width, self.input_c_dim]
    image_depth_semantic_dims = [self.output_height, self.output_width, self.output_c_dim]
    image_semantic_dims = [self.output_height, self.output_width, 1]
    image_depth_dims = [self.output_height, self.output_width, 1]

    #input of generator
    self.real_rgb_images = tf.placeholder(
      tf.float32, [self.batch_size] + image_rgb_dims, name='real_rgb_images')
    #input of discriminator
    self.real_depth_semantic_images = tf.placeholder(
      tf.float32, [self.batch_size] + image_depth_semantic_dims, name='real_depth_semantic_images')
    self.real_semantic_images = tf.placeholder(
      tf.float32, [self.batch_size] + image_semantic_dims, name='real_semantic_images')
    self.real_depth_images = tf.placeholder(
      tf.float32, [self.batch_size] + image_depth_dims, name='real_depth_images')

    #generator, concatenate depth and semantics
    self.G1, self.G2 = self.generator(self.real_rgb_images, name = 'G')
    #self.G = tf.concat([self.G1, self.G2], 3)
    self.G = tf.concat([self.G1, tf.cast(tf.expand_dims(tf.argmax(self.G2, axis = 3), axis = 3), tf.float32)], 3)

    #input real image pairs
    self.D, self.D_logits = self.discriminator(self.real_depth_semantic_images, reuse = False, name = 'D')
    #input fake image pairs
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse = True, name = 'D')
    
    #generate samples
    self.sampler1, self.sampler2 = self.sampler(self.real_rgb_images, name = 'sampler')


    self.d_sum = tf.summary.histogram("d", self.D)
    self.d__sum = tf.summary.histogram("d_", self.D_)
    # self.g_sum = tf.summary.image("g", self.G)


    soft_max_label = tf.random_uniform(shape = [1], minval = 0.75, maxval = 1)
    soft_min_label = tf.random_uniform(shape = [1], minval = 0, maxval = 0.25)

    #discriminator loss
    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(self.D)*soft_max_label, logits=self.D_logits))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(self.D_)*soft_min_label, logits=self.D_logits_))
    self.d_loss = self.d_loss_real + self.d_loss_fake

    #generator loss
    self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(self.D_)*soft_max_label, logits=self.D_logits_)) \
    + self.L1_lambda * ( tf.reduce_mean(tf.abs(self.real_depth_images - self.G1))) \
    + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(tf.cast(self.real_semantic_images, tf.int32) - 1, [-1, self.crop_height * self.crop_width]),
    logits = tf.reshape(self.G2, [-1,self.crop_height * self.crop_width, 40])))

    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

    #Trainable variables
    t_vars = tf.trainable_variables()
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    # print("Generator variable {}".format([v.op.name for v in self.g_vars]))
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    # print("Discriminator variable {}".format([v.op.name for v in self.d_vars]))
    self.saver = tf.train.Saver()



  def sample_model(self, data, sample_dir, epoch, idx, config):
    RGB_sample_images = get_RGB_batch(batch_file = data, 
                                      start_idx = idx*config.batch_size, 
                                      end_idx = (idx+1)*config.batch_size, 
                                      input_height = self.input_height, 
                                      input_width = self.input_width, 
                                      resize_height = self.crop_height, 
                                      resize_width = self.crop_width, 
                                      is_crop = False)

    Depth_sample_images = get_depth_batch(batch_file = data, 
                                      start_idx = idx*config.batch_size, 
                                      end_idx = (idx+1)*config.batch_size, 
                                      input_height = self.input_height, 
                                      input_width = self.input_width, 
                                      resize_height = self.crop_height, 
                                      resize_width = self.crop_width, 
                                      is_crop = False)

    Semantic_sample_images = get_semantic_batch(batch_file = data, 
                                      start_idx = idx*config.batch_size, 
                                      end_idx = (idx+1)*config.batch_size, 
                                      input_height = self.input_height, 
                                      input_width = self.input_width, 
                                      resize_height = self.crop_height, 
                                      resize_width = self.crop_width, 
                                      is_crop = False)
    Depth_semantic_sample_images = np.concatenate((Depth_sample_images, Semantic_sample_images),axis = 3)

    samples1, samples2, d_loss, g_loss = self.sess.run(
        [self.sampler1, self.sampler2, self.d_loss, self.g_loss],
        feed_dict = {self.real_rgb_images: RGB_sample_images, 
                     self.real_depth_semantic_images: Depth_semantic_sample_images,
                     self.real_depth_images: Depth_sample_images,
                     self.real_semantic_images: Semantic_sample_images})


    save_images(samples1, [self.batch_size, 1],
                './{}/train_{:02d}_{:04d}_depth.png'.format(sample_dir, epoch, idx))
    save_images(np.argmax(samples2, axis = 3), [self.batch_size, 1],
                './{}/train_{:02d}_{:04d}_semantic.png'.format(sample_dir, epoch, idx))
    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))


  def train(self, config):
    """Train CoGAN"""
    #load mat file
    data = load_mat("../nyu_depth_v2_labeled.mat")


    #Optimizer for generator and discriminator
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.momentum1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.momentum1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    
    tf.global_variables_initializer().run()

    self.g_sum = tf.summary.merge([self.d__sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    counter = 1
    start_time = time.time()

    #load checkpoint
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    #compute batch idx
    batch_idxs = 1400 // config.batch_size

    for epoch in xrange(config.epoch):
      for idx in xrange(0, batch_idxs):


        #prepare for input
        RGB_batch_images = get_RGB_batch(batch_file = data, 
                                          start_idx = idx*config.batch_size, 
                                          end_idx = (idx+1)*config.batch_size, 
                                          input_height = self.input_height, 
                                          input_width = self.input_width, 
                                          resize_height = self.crop_height, 
                                          resize_width = self.crop_width, 
                                          is_crop = False)

        Depth_batch_images = get_depth_batch(batch_file = data, 
                                          start_idx = idx*config.batch_size, 
                                          end_idx = (idx+1)*config.batch_size, 
                                          input_height = self.input_height, 
                                          input_width = self.input_width, 
                                          resize_height = self.crop_height, 
                                          resize_width = self.crop_width, 
                                          is_crop = False)

        Semantic_batch_images = get_semantic_batch(batch_file = data, 
                                          start_idx = idx*config.batch_size, 
                                          end_idx = (idx+1)*config.batch_size, 
                                          input_height = self.input_height, 
                                          input_width = self.input_width, 
                                          resize_height = self.crop_height, 
                                          resize_width = self.crop_width, 
                                          is_crop = False)

        Depth_semantic_batch_images = np.concatenate((Depth_batch_images, Semantic_batch_images),axis = 3)


        # Update Discriminator
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ self.real_rgb_images: RGB_batch_images, 
                      self.real_depth_semantic_images: Depth_semantic_batch_images})
        self.writer.add_summary(summary_str, counter)

        # Update Generator twice  
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.real_rgb_images: RGB_batch_images, 
                      self.real_depth_semantic_images: Depth_semantic_batch_images,
                      self.real_depth_images: Depth_batch_images,
                      self.real_semantic_images: Semantic_batch_images})
        self.writer.add_summary(summary_str, counter)

        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.real_rgb_images: RGB_batch_images, 
                     self.real_depth_semantic_images: Depth_semantic_batch_images,
                     self.real_depth_images: Depth_batch_images,
                     self.real_semantic_images: Semantic_batch_images})
        self.writer.add_summary(summary_str, counter)
        
        errD = self.d_loss.eval({self.real_rgb_images: RGB_batch_images, self.real_depth_semantic_images: Depth_semantic_batch_images})
        errG = self.g_loss.eval({self.real_rgb_images: RGB_batch_images, 
                                 self.real_depth_semantic_images: Depth_semantic_batch_images,
                                 self.real_depth_images: Depth_batch_images,
                                 self.real_semantic_images: Semantic_batch_images})

        #print loss
        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD, errG))

        #print test images
        if np.mod(counter, 100) == 1:
          self.sample_model(data, self.sample_dir, epoch, idx, config)

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)


  def discriminator(self, image, reuse = False, name = 'D'):
    # image is 256 x 256 x output_c_dim
    with tf.variable_scope("D") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        # h3 is (16 x 16 x self.df_dim*8)
        h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*8, name='d_h4_conv')))
        # h3 is (8 x 8 x self.df_dim*8)
        h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim*8, name='d_h5_conv')))
        # h3 is (4 x 4 x self.df_dim*8)
        h6 = conv2d(h5, 1, d_h = 1, d_w = 1, name='d_h6_conv')
        #h6 is (4*4*1)
        h6 = tf.reshape(h6,[self.batch_size,-1])
        return tf.nn.sigmoid(h6), h6

      



  def generator(self, image, name = 'G'):

    with tf.variable_scope("G") as scope:
      #256
      s_h, s_w = self.output_height, self.output_width
      #128
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      #64
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      #32
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      #16
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
      #8
      s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
      #4
      s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)
      #2
      s_h128, s_w128 = conv_out_size_same(s_h64, 2), conv_out_size_same(s_w64, 2)

      # image is (256 x 256 x input_c_dim)
      e1 = lrelu(conv2d(image, self.gf_dim, name='g_e1_conv'))
      # e1 is (128 x 128 x self.gf_dim)
      e2 = lrelu(self.g_bn_e2(conv2d(e1, self.gf_dim*2, name='g_e2_conv')))
      # e2 is (64 x 64 x self.gf_dim*2)
      e3 = lrelu(self.g_bn_e3(conv2d(e2, self.gf_dim*4, name='g_e3_conv')))
      # e3 is (32 x 32 x self.gf_dim*4)
      e4 = lrelu(self.g_bn_e4(conv2d(e3, self.gf_dim*8, name='g_e4_conv')))
      # e4 is (16 x 16 x self.gf_dim*8)
      e5 = lrelu(self.g_bn_e5(conv2d(e4, self.gf_dim*8, name='g_e5_conv')))
      # e5 is (8 x 8 x self.gf_dim*8)
      e6 = lrelu(self.g_bn_e6(conv2d(e5, self.gf_dim*8, name='g_e6_conv')))
      # e6 is (4 x 4 x self.gf_dim*8)
      e7 = lrelu(self.g_bn_e7(conv2d(e6, self.gf_dim*8, name='g_e7_conv')))
      # e7 is (2 x 2 x self.gf_dim*8)
      e8 = lrelu(self.g_bn_e8(conv2d(e7, self.gf_dim*8, name='g_e8_conv')))
      # e8 is (1 x 1 x self.gf_dim*8)

      self.d1, self.d1_w, self.d1_b = deconv2d(e8,[self.batch_size, s_h128, s_w128, self.gf_dim*8], name='g_d1', with_w=True)
      d1 = tf.nn.relu(tf.nn.dropout(self.g_bn_d1(self.d1), 0.5))
      d1 = tf.concat([d1, e7], 3)
      # d1 is (2 x 2 x self.gf_dim*8*2)

      self.d2, self.d2_w, self.d2_b = deconv2d(d1,[self.batch_size, s_h64, s_w64, self.gf_dim*8], name='g_d2', with_w=True)
      d2 = tf.nn.relu(tf.nn.dropout(self.g_bn_d2(self.d2), 0.5))
      d2 = tf.concat([d2, e6], 3)
      # d2 is (4 x 4 x self.gf_dim*8*2)

      self.d3, self.d3_w, self.d3_b = deconv2d(d2,[self.batch_size, s_h32, s_w32, self.gf_dim*8], name='g_d3', with_w=True)
      d3 = tf.nn.relu(tf.nn.dropout(self.g_bn_d3(self.d3), 0.5))
      d3 = tf.concat([d3, e5], 3)
      # d3 is (8 x 8 x self.gf_dim*8*2)

      self.d4, self.d4_w, self.d4_b = deconv2d(d3,[self.batch_size, s_h16, s_w16, self.gf_dim*8], name='g_d4', with_w=True)
      d4 = tf.nn.relu(self.g_bn_d4(self.d4))
      d4 = tf.concat([d4, e4], 3)
      # d4 is (16 x 16 x self.gf_dim*8*2)

      self.d5, self.d5_w, self.d5_b = deconv2d(d4,[self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_d5', with_w=True)
      d5 = tf.nn.relu(self.g_bn_d5(self.d5))
      d5 = tf.concat([d5, e3], 3)
      # d5 is (32 x 32 x self.gf_dim*4*2)

      self.d6, self.d6_w, self.d6_b = deconv2d(d5,[self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_d6', with_w=True)
      d6 = tf.nn.relu(self.g_bn_d6(self.d6))
      d6 = tf.concat([d6, e2], 3)
      # d6 is (64 x 64 x self.gf_dim*2*2)

      self.d7, self.d7_w, self.d7_b = deconv2d(d6,[self.batch_size, s_h2, s_w2, self.gf_dim], name='g_d7', with_w=True)
      d7 = tf.nn.relu(self.g_bn_d7(self.d7))
      d7 = tf.concat([d7, e1], 3)
      # d7 is (128 x 128 x self.gf_dim*1*2)

      #domain 1
      self.d8_1, self.d8_1_w, self.d8_1_b = deconv2d(d7,[self.batch_size, s_h2, s_w2, self.gf_dim], d_h=1, d_w=1, name='g_d8_1', with_w=True)
      d8_1 = tf.nn.relu(self.g_bn_d8_1(self.d8_1))
      # d8_1 is (128 x 128 x self.gf_dim)
      self.d9_1, self.d9_1_w, self.d9_1_b = deconv2d(d8_1,[self.batch_size, s_h2, s_w2, 32], d_h=1, d_w=1,name='g_d9_1', with_w=True)
      d9_1 = tf.nn.relu(self.g_bn_d9_1(self.d9_1))
      # d9_1 is (128 x 128 x self.gf_dim/2)
      self.d10_1, self.d10_1_w, self.d10_1_b = deconv2d(d9_1,[self.batch_size, s_h, s_w, 1], name='g_d10_1', with_w=True)
      d10_1 = tf.nn.tanh(self.d10_1)
      # d10_1 is (256 x 256 x 1) : depth

      #domain 2
      self.d8_2, self.d8_2_w, self.d8_2_b = deconv2d(d7,[self.batch_size, s_h2, s_w2, self.gf_dim], d_h=1, d_w=1,name='g_d8_2', with_w=True)
      d8_2 = tf.nn.relu(self.g_bn_d8_2(self.d8_2))
      # d8_2 is (128 x 128 x self.gf_dim)
      self.d9_2, self.d9_2_w, self.d9_2_b = deconv2d(d8_2,[self.batch_size, s_h2, s_w2, 32], d_h=1, d_w=1,name='g_d9_2', with_w=True)
      d9_2 = tf.nn.relu(self.g_bn_d9_2(self.d9_2))
      # d9_2 is (128 x 128 x self.gf_dim/2)
      self.d10_2, self.d10_2_w, self.d10_2_b = deconv2d(d9_2,[self.batch_size, s_h, s_w, 40], name='g_d10_2', with_w=True)
      d10_2 = tf.nn.tanh(self.d10_2)
      # d10_2 is (256 x 256 x 40) : semantic

      return d10_1, d10_2



  def sampler(self, image, name = None):
    with tf.variable_scope("G") as scope:
      tf.get_variable_scope().reuse_variables()
      #256
      s_h, s_w = self.output_height, self.output_width
      #128
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      #64
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      #32
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      #16
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
      #8
      s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
      #4
      s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)
      #2
      s_h128, s_w128 = conv_out_size_same(s_h64, 2), conv_out_size_same(s_w64, 2)

      # image is (256 x 256 x input_c_dim)
      e1 = lrelu(conv2d(image, self.gf_dim, name='g_e1_conv'))
      # e1 is (128 x 128 x self.gf_dim)
      e2 = lrelu(self.g_bn_e2(conv2d(e1, self.gf_dim*2, name='g_e2_conv')))
      # e2 is (64 x 64 x self.gf_dim*2)
      e3 = lrelu(self.g_bn_e3(conv2d(e2, self.gf_dim*4, name='g_e3_conv')))
      # e3 is (32 x 32 x self.gf_dim*4)
      e4 = lrelu(self.g_bn_e4(conv2d(e3, self.gf_dim*8, name='g_e4_conv')))
      # e4 is (16 x 16 x self.gf_dim*8)
      e5 = lrelu(self.g_bn_e5(conv2d(e4, self.gf_dim*8, name='g_e5_conv')))
      # e5 is (8 x 8 x self.gf_dim*8)
      e6 = lrelu(self.g_bn_e6(conv2d(e5, self.gf_dim*8, name='g_e6_conv')))
      # e6 is (4 x 4 x self.gf_dim*8)
      e7 = lrelu(self.g_bn_e7(conv2d(e6, self.gf_dim*8, name='g_e7_conv')))
      # e7 is (2 x 2 x self.gf_dim*8)
      e8 = lrelu(self.g_bn_e8(conv2d(e7, self.gf_dim*8, name='g_e8_conv')))
      # e8 is (1 x 1 x self.gf_dim*8)

      self.d1, self.d1_w, self.d1_b = deconv2d(e8,
          [self.batch_size, s_h128, s_w128, self.gf_dim*8], name='g_d1', with_w=True)
      d1 = tf.nn.relu(tf.nn.dropout(self.g_bn_d1(self.d1), 0.5))
      d1 = tf.concat([d1, e7], 3)
      # d1 is (2 x 2 x self.gf_dim*8*2)

      self.d2, self.d2_w, self.d2_b = deconv2d(d1,
          [self.batch_size, s_h64, s_w64, self.gf_dim*8], name='g_d2', with_w=True)
      d2 = tf.nn.relu(tf.nn.dropout(self.g_bn_d2(self.d2), 0.5))
      d2 = tf.concat([d2, e6], 3)
      # d2 is (4 x 4 x self.gf_dim*8*2)

      self.d3, self.d3_w, self.d3_b = deconv2d(d2,
          [self.batch_size, s_h32, s_w32, self.gf_dim*8], name='g_d3', with_w=True)
      d3 = tf.nn.relu(tf.nn.dropout(self.g_bn_d3(self.d3), 0.5))
      d3 = tf.concat([d3, e5], 3)
      # d3 is (8 x 8 x self.gf_dim*8*2)

      self.d4, self.d4_w, self.d4_b = deconv2d(d3,
          [self.batch_size, s_h16, s_w16, self.gf_dim*8], name='g_d4', with_w=True)
      d4 = tf.nn.relu(self.g_bn_d4(self.d4))
      d4 = tf.concat([d4, e4], 3)
      # d4 is (16 x 16 x self.gf_dim*8*2)

      self.d5, self.d5_w, self.d5_b = deconv2d(d4,
          [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_d5', with_w=True)
      d5 = tf.nn.relu(self.g_bn_d5(self.d5))
      d5 = tf.concat([d5, e3], 3)
      # d5 is (32 x 32 x self.gf_dim*4*2)

      self.d6, self.d6_w, self.d6_b = deconv2d(d5,
          [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_d6', with_w=True)
      d6 = tf.nn.relu(self.g_bn_d6(self.d6))
      d6 = tf.concat([d6, e2], 3)
      # d6 is (64 x 64 x self.gf_dim*2*2)

      self.d7, self.d7_w, self.d7_b = deconv2d(d6,
          [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_d7', with_w=True)
      d7 = tf.nn.relu(self.g_bn_d7(self.d7))
      d7 = tf.concat([d7, e1], 3)
      # d7 is (128 x 128 x self.gf_dim*1*2)

      #domain 1
      self.d8_1, self.d8_1_w, self.d8_1_b = deconv2d(d7,
          [self.batch_size, s_h2, s_w2, self.gf_dim], d_h=1, d_w=1,name='g_d8_1', with_w=True)
      d8_1 = tf.nn.relu(self.g_bn_d8_1(self.d8_1))
      # d8_1 is (128 x 128 x self.gf_dim)
      self.d9_1, self.d9_1_w, self.d9_1_b = deconv2d(d8_1,
          [self.batch_size, s_h2, s_w2, 32], d_h=1, d_w=1,name='g_d9_1', with_w=True)
      d9_1 = tf.nn.relu(self.g_bn_d9_1(self.d9_1))
      # d9_1 is (128 x 128 x self.gf_dim/2)
      self.d10_1, self.d10_1_w, self.d10_1_b = deconv2d(d9_1,
          [self.batch_size, s_h, s_w, 1], name='g_d10_1', with_w=True)
      d10_1 = tf.nn.tanh(self.d10_1)
      # d10_1 is (256 x 256 x 1) : depth

      #domain 2
      self.d8_2, self.d8_2_w, self.d8_2_b = deconv2d(d7,
          [self.batch_size, s_h2, s_w2, self.gf_dim], d_h=1, d_w=1,name='g_d8_2', with_w=True)
      d8_2 = tf.nn.relu(self.g_bn_d8_2(self.d8_2))
      # d8_2 is (128 x 128 x self.gf_dim)
      self.d9_2, self.d9_2_w, self.d9_2_b = deconv2d(d8_2,
          [self.batch_size, s_h2, s_w2, 32], d_h=1, d_w=1,name='g_d9_2', with_w=True)
      d9_2 = tf.nn.relu(self.g_bn_d9_2(self.d9_2))
      # d9_2 is (128 x 128 x self.gf_dim/2)
      self.d10_2, self.d10_2_w, self.d10_2_b = deconv2d(d9_2,
          [self.batch_size, s_h, s_w, 40], name='g_d10_2', with_w=True)
      d10_2 = tf.nn.tanh(self.d10_2)
      # d10_2 is (256 x 256 x 40) : semantic

      return d10_1, d10_2
      

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "CoGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False
