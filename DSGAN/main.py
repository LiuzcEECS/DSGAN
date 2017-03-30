import os
import scipy.misc
import numpy as np

from model import CoGAN
from utils import pp

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [200]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("L1_lambda", 50, "parameter for L1 loss [50]")
flags.DEFINE_float("momentum1", 0.5, "1st momentum parameter [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [1]")
flags.DEFINE_integer("input_height", 480, "The size of the input images [480]")
flags.DEFINE_integer("input_width", 640, "The size of the input images [640]")
flags.DEFINE_integer("crop_height", 256, "The size of the input images after cropping [256]")
flags.DEFINE_integer("crop_width", None, "The size of the input images after cropping. If None, same value as output_height [None]")
flags.DEFINE_integer("output_height", 256, "The size of the output images to produce [256]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("input_c_dim", 3, "Channels of input images [3]")
flags.DEFINE_integer("output_c_dim", 2, "Channels of output images [2]")
flags.DEFINE_integer("df_dim", 64, "Channels of first discriminator conv layer [64]")
flags.DEFINE_integer("gf_dim", 64, "Channels of first generator conv layer [64]")
flags.DEFINE_string("dataset", "NYU_Depth", "The name of dataset [NYU_Depth]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the test images [tests]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height
  if FLAGS.crop_width is None:
    FLAGS.crop_width = FLAGS.crop_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  if not os.path.exists(FLAGS.test_dir):
    os.makedirs(FLAGS.test_dir)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    
    Cogan = CoGAN(
        sess,
        L1_lambda = FLAGS.L1_lambda,
        batch_size=FLAGS.batch_size,
        input_width = FLAGS.input_width,
        input_height = FLAGS.input_height,
        crop_width = FLAGS.crop_width,
        crop_height = FLAGS.crop_width,
        input_c_dim = FLAGS.input_c_dim,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        output_c_dim = FLAGS.output_c_dim,
        gf_dim = FLAGS.gf_dim,
        df_dim = FLAGS.df_dim,
        dataset_name=FLAGS.dataset,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        is_crop=FLAGS.is_crop)

    if FLAGS.is_train:
      Cogan.train(FLAGS)
    else:
      if not Cogan.load(FLAGS.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")
      # Cogan.test(FLAGS)

if __name__ == '__main__':
  tf.app.run()
