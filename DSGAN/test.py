import os
import numpy as np
from scipy import misc
from scipy.io import savemat
import utils
import cv2
from dataset_defs import NYUDepthModelDefs as data_def

DEPTH_DIR = "/media/2T/data/nyu_depth_v2_labeled.mat"
def vis(image):
    misc.imshow(np.squeeze(image))

def merge_mat():
    mat_dir = "/media/2T/data/data_40/benchmarkData/metadata/classMapping40.mat"
    f = utils.load_mat(mat_dir)
    lut = f["mapClass"][0]
    lut = np.concatenate((np.array([40]),lut))
    print lut

def test_utils():

    print("Test Utils")
    num = 1
    mat_file = utils.load_mat(DEPTH_DIR)
    RGB_list = utils.get_RGB_batch(mat_file,0,9,480,640,240,320,False)
    vis(RGB_list[0])
    depth_list = utils.get_depth_batch(mat_file,0,9,480,640,240,320,False)
    vis(utils.color_depth(np.array([depth_list[num]], dtype = "float")))
    semantic_list = utils.get_semantic_batch(mat_file,0,9,480,640,240,320,False)

    vis(utils.color_semantic(np.array([semantic_list[num]], dtype = "float")))

def main():
    test_utils()

if __name__ == "__main__":
    main()
