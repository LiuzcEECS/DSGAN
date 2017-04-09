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

def save_images(images, sample_dir, epoch, idx, label, is_grayscale = False):
  if is_grayscale == True:
    img = np.zeros((images.shape[1], images.shape[2]))
    for i in range(images.shape[0]):
      img = images[i].reshape((images.shape[1],images.shape[2]))
      scipy.misc.imsave('./{}/train_{:02d}_{:04d}_{:02d}_{}.png'.format(sample_dir, epoch, idx, i, label), img)
  else:
    img = np.zeros((images.shape[1], images.shape[2],3))
    for i in range(images.shape[0]):
      img = images[i]
      scipy.misc.imsave('./{}/train_{:02d}_{:04d}_{:02d}_{}.png'.format(sample_dir, epoch, idx, i, label), img)
    # return imsave(images, size, image_path, is_grayscale)


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
    tar = np.exp(img*data_def().logdepths_std + data_def().logdepths_mean)
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

def depth_resize_trans(images,config):
  images = np.exp(images*data_def().logdepths_std + data_def().logdepths_mean)
  res = []
  for i in range(images.shape[0]):
    # print images[i].shape
    tem = (misc.imresize(images[i], (config.input_height, config.input_width), "nearest",'F'))
    res.append(tem)
  return np.array(res).astype(np.float)[:,:,:,None]

def rel_error(ground_truth, predicted):
  return np.mean(np.absolute(ground_truth - predicted) / ground_truth)

def rel_sqr_error(ground_truth, predicted):
  return np.mean(np.power(ground_truth - predicted,2) / ground_truth)

def log10_error(ground_truth,predicted):
  return np.mean(np.absolute(np.log10(ground_truth) - np.log10(predicted)))


def rms_linear_error(ground_truth, predicted):
  return np.sqrt(np.mean(np.power(ground_truth-predicted,2)))

def rms_log_error(ground_truth,predicted):
  return np.sqrt(np.mean(np.power(np.log(ground_truth) - np.log(predicted),2)))

def thr_accuracy(ground_truth, predicted, threshold):
  tmp = np.maximum(ground_truth / predicted , predicted / ground_truth)
  flag = (tmp < threshold).astype(int)
  return np.count_nonzero(flag) / predicted.size



def depth_error(ground_truth, predicted, config):
  ground_truth = depth_resize_trans(ground_truth,config)
  predicted = depth_resize_trans(predicted,config)

  rel = rel_error(ground_truth, predicted)
  rel_sqr = rel_sqr_error(ground_truth, predicted)
  log10 = log10_error(ground_truth,predicted)
  rms_linear = rms_linear_error(ground_truth, predicted)
  rms_log = rms_log_error(ground_truth,predicted)
  thr_1 = thr_accuracy(ground_truth, predicted, 1.25)
  thr_2 = thr_accuracy(ground_truth, predicted, 1.25*1.25)
  thr_3 = thr_accuracy(ground_truth, predicted, 1.25*1.25*1.25)
  print("[Sample] rel: {:.8f}, rel_sqr: {:.8f}, log10: {:.8f}".format(rel,rel_sqr,log10))
  print("[Sample] rms_linear: {:.8f}, rms_log: {:.8f}".format(rms_linear,rms_log))
  print("[Sample] thr_1: {:.8f}, thr_2: {:.8f}, thr_3: {:.8f}".format(thr_1,thr_2,thr_3))

def semantic_pixel_acc(labels, results):
    cnt = 0.0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                if labels[i][j][k][0] == results[i][j][k][0]:
                    cnt += 1.0
    cnt = cnt / labels.shape[0] / labels.shape[1] / labels.shape[2]
    return cnt

def semantic_class_acc(labels, results): # or mean accuracy ?
    cnt_class = np.zeros((41), dtype = np.float32)
    cnt_cor = np.zeros((41), dtype = np.float32)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                if labels[i][j][k][0] == results[i][j][k][0]:
                    cnt_cor[results[i][j][k][0]] += 1
                cnt_class[labels[i][j][k][0]] += 1
    s = 0.0
    for i in range(1,41):
        if cnt_cor[i] == 0:
            continue
        s += float(cnt_cor[i] / cnt_class[i])
    s = s / 40.0

    return s

def semantic_mean_IoU_acc(labels, results):
    cnt_u = np.zeros((41), dtype = np.float32)
    cnt_cor = np.zeros((41), dtype = np.float32)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                if labels[i][j][k][0] == results[i][j][k][0]:
                    cnt_cor[results[i][j][k][0]] += 1
                cnt_u[labels[i][j][k][0]] += 1
                cnt_u[results[i][j][k][0]] += 1
    s = 0.0
    for i in range(1,41):
        if cnt_cor[i] == 0:
            continue
        s += float(cnt_cor[i] / (cnt_u[i] - cnt_cor[i]))

    s = s / 40.0
    return s


