"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import colorsys
from matplotlib import cm
import json
import random
import pprint
import scipy.misc
from scipy import misc
from scipy.io import loadmat
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import h5py
from dataset_defs import NYUDepthModelDefs as data_def

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=True, is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, is_crop, is_grayscale = is_grayscale)

def save_images(images, size, image_path, is_grayscale = False):
    return imsave(inverse_transform(images), size, image_path, is_grayscale)
    # return imsave(images, size, image_path, is_grayscale)

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge(images, size, is_grayscale):
  h, w = images.shape[1], images.shape[2]
  if is_grayscale == True:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      image = image.reshape((h,w))
      img[j*h:j*h+h, i*w:i*w+w] = image
  else:
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path, is_grayscale):
  return scipy.misc.imsave(path, merge(images, size, is_grayscale))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height=64, input_width=64, 
              resize_height=64, resize_width=64, is_crop= False, is_grayscale=False):
  if is_crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  if is_grayscale:
    return np.array(cropped_image)/32767.5 - 1.
  else:
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.


'''
    Preprocessing of NYU Depth Dataset
'''
LUT = [40,40,40,3,22,5,40,12,38,40,40,2,39,40,40,26,40,24,40,7,40,1,40,40,34,
38,29,40,8,40,40,40,40,38,40,40,14,40,38,40,40,40,15,39,40,30,40,40,39,40,
39,38,40,38,40,37,40,38,38,9,40,40,38,40,11,38,40,40,40,40,40,40,40,40,40,
40,40,40,40,38,13,40,40,6,40,23,40,39,10,16,40,40,40,40,38,40,40,40,40,40,
40,40,40,40,38,40,39,40,40,40,40,39,38,40,40,40,40,40,40,18,40,40,19,28,33,
40,40,40,40,40,40,40,40,40,38,27,36,40,40,40,40,21,40,20,35,40,40,40,40,40,
40,40,40,38,40,40,40,4,32,40,40,39,40,39,40,40,40,40,40,17,40,40,25,40,39,
40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,
40,40,40,40,39,40,40,40,40,40,40,40,40,40,39,38,38,40,40,39,40,39,40,38,39,
38,40,40,40,40,40,40,40,40,40,40,39,40,38,40,40,38,38,40,40,40,40,40,40,40,
40,40,40,40,40,40,38,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,
40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,39,40,40,40,38,40,40,39,40,
40,38,40,40,40,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,
40,40,40,40,40,40,31,40,40,40,40,40,40,40,38,40,40,38,39,39,40,40,40,40,40,
40,40,40,40,38,40,39,40,40,39,40,40,40,38,40,40,40,40,40,40,40,40,38,39,40,
40,40,40,40,40,38,40,40,40,40,40,40,40,40,40,40,40,38,39,40,40,40,40,40,40,
40,39,40,40,40,40,40,40,38,40,40,40,38,40,39,40,40,40,39,39,40,40,40,40,40,
40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,39,39,40,40,39,39,40,
40,40,40,38,40,40,38,39,39,40,39,40,39,38,40,40,40,40,40,40,40,40,40,40,40,
39,40,38,40,39,40,40,40,40,40,39,39,40,40,40,40,40,40,39,39,40,40,38,39,39,
40,40,40,40,40,40,40,40,40,39,39,40,40,40,40,39,40,40,40,40,40,39,40,40,39,
40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,38,40,40,40,40,40,40,40,39,
38,39,40,38,39,40,39,40,39,40,40,40,40,40,40,40,40,38,40,40,40,40,40,38,40,
40,39,40,40,40,39,40,38,40,40,40,40,40,40,40,40,38,40,40,40,40,40,40,40,40,
40,40,40,40,40,40,40,40,39,38,40,40,38,40,40,38,40,40,40,40,40,40,40,40,40,
39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,38,40,40,38,40,40,40,40,
40,40,40,40,40,40,40,38,38,38,40,40,40,38,40,40,40,38,38,40,40,40,40,40,40,
40,40,40,40,40,40,40,40,40,40,40,38,40,40,40,40,40,40,40,40,40,40,40,40,40,
38,40,38,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,
40,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,
40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,40,39,40,40,
40,40,38,38,40,40,40,38,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,40,40,
39,40,40,39,39,40,40,40,40,40,40,40,40,39,39,39,40,40,40,40,39,40,40,40,40,
40,40,40,40,39,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,
38,40,40,40,40,40,40,40,39,40,40,38,40,39,40,40,40,40,38,40,40,40,40,40,38,
40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,39,40,40]

LUT = np.array(LUT)

def load_mat(batch_dir):
    try:
        f = h5py.File(batch_dir)
    except:
        f = loadmat(batch_dir)
    return f

def get_RGB_batch(batch_file, start_idx, end_idx, input_height, input_width, resize_height, resize_width, is_crop = False):
    seq = np.swapaxes(np.array(batch_file["images"][start_idx:end_idx]),1 ,3 )
    # print seq.shape
    assert input_height == seq.shape[1] and input_width == seq.shape[2], "size mismatch"
    res = []
    for i in range(seq.shape[0]):
        res.append(misc.imresize(seq[i], (resize_height, resize_width, 3), "nearest"))
    return res

def get_depth_batch(batch_file, start_idx, end_idx, input_height, input_width, resize_height, resize_width, is_crop = False):
    seq = np.swapaxes(np.array(batch_file["depths"][start_idx:end_idx]),1 ,2 )
    # print seq.shape
    assert input_height == seq.shape[1] and input_width == seq.shape[2], "size mismatch"
    res = []
    opt = data_def()
    for i in range(seq.shape[0]):
        tem = (misc.imresize(seq[i], (resize_height, resize_width), "nearest",'F'))
        t = tem.astype(float) + 1e-4
        res.append(np.log(t))
        res[i] -= opt.logdepths_mean
        res[i] /= opt.logdepths_std

    return np.array(res)[:,:,:,None]

def get_semantic_batch(batch_file, start_idx, end_idx, input_height, input_width, resize_height, resize_width, is_crop = False):
    seq = np.swapaxes(np.array(batch_file["labels"][start_idx:end_idx]),1 ,2 )
    # print seq.shape
    assert input_height == seq.shape[1] and input_width == seq.shape[2], "size mismatch"
    res = []
    for i in range(seq.shape[0]):
        res.append(misc.imresize(seq[i], (resize_height, resize_width), "nearest",'F'))
    return LUT[np.array(res,dtype = 'int16')][:,:,:,None]

def divide_output(array):
    res_depth, res_semantic = np.spilt(array,[3],3)
    return res_depth, res_semantic

'''
    Visualization
'''
def depth_trans(img):
    tar = np.exp(img*data_def().logdepths_std)
    maxx = np.max(tar)
    minn = np.min(tar)

    if img.ndim == 4:
        for i in range(tar.shape[0]):
            for j in range(tar.shape[1]):
                for k in range(tar.shape[2]):
                    tar[i][j][k] = (float(tar[i][j][k]) - minn)/(maxx - minn) * 255.0
    else:
        for i in range(tar.shape[0]):
            for j in range(tar.shape[1]):
                tar[i][j] = (float(tar[i][j]) - minn)/(maxx - minn) * 255.0

    return tar

def color_depth(depth_m):
    depth_mat = depth_trans(depth_m)
    res = np.zeros(np.concatenate((depth_mat.shape[:-1],[3])))
    if res.ndim == 4:
        for i in range(depth_mat.shape[0]):
            res[i] = cm.jet(depth_mat[i][:,:,0]/255.0)[:,:,:3]
    else:
        res = cm.jet(depth_mat[:,:,0]/255.0)[:,:,:3]

    return res



def color_semantic(semantic_mat):
    res = np.zeros(np.concatenate((semantic_mat.shape[:-1],[3])))
    if res.ndim == 4:
        for i in range(semantic_mat.shape[0]):
            class_num = len(np.unique(semantic_mat[i].flatten()))
            color_dict = {}
            for num,c in enumerate(np.unique(semantic_mat[i].flatten())):
                color_dict[c] = colorsys.hsv_to_rgb(1.0 / class_num * num,1.0,1.0);
            color_dict[0] = [0,0,0]
            for j in range(semantic_mat.shape[1]):
                for k in range(semantic_mat.shape[2]):
                    res[i][j][k] = color_dict[semantic_mat[i][j][k][0]]
    else:
        class_num = len(np.unique(semantic_mat.flatten()))
        color_dict = {}
        for num,c in enumerate(np.unique(semantic_mat.flatten())):
            color_dict[c] = colorsys.hsv_to_rgb(1.0 / class_num * num,1.0,1.0);
        color_dict[0] = [0,0,0]
        for j in range(semantic_mat.shape[0]):
            for k in range(semantic_mat.shape[1]):
                res[j][k] = color_dict[semantic_mat[j][k][0]]

    return res
