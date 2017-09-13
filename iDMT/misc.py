# Common imports and miscellaneous functions
# DMT is originally by: Jedrzej
# Author: Kemal Berk Kocabagli 

import os,sys
os.environ['GLOG_minloglevel'] = '2'  # suppress Caffe logs

import numpy as np # for mathematical operations
import glob # for paths
import cv2 # for operations on images
import random # for random subsampling
import time # to measure function speeds
import scipy.spatial as sp # to calculate cosine similarity

import pickle # a module for serialization/de-serialization of Python objects
# more details at: https://docs.python.org/2/library/pickle.html

# scales and displays an image 
def displayImage(PATH, img, scale):
	if(img.shape[2]>3): img = img.transpose((1,2,0))
	img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale))) # resize
	cv2.imshow(PATH,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# reads and displays a pickle, which is a data storage format
def readAndDisplayPickle(PATH):
	print("Reading image dictionary...")
	with(open(PATH, "rb")) as openfile:
		print(pickle.load(openfile))