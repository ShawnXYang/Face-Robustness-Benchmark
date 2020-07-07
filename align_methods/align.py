from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
#import facenet
import detect_face
import random
from time import sleep
import face_image
import face_preprocess
from skimage import transform as trans
import cv2

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe,GTframe):
  x1 = Reframe[0];
  y1 = Reframe[1];
  width1 = Reframe[2]-Reframe[0];
  height1 = Reframe[3]-Reframe[1];

  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]

  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Graph().as_default():
    sess = tf.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

#@profile
def align(img, image_size=(112,112)):
    minsize = 20
    threshold = [0.6,0.7,0.9]
    factor = 0.85

    if img.ndim<2:
       print('Image dim error' % image_path)
    if img.ndim == 2:
       img = to_rgb(img)
       img = img[:,:,0:3]
    _minsize = minsize
    _bbox = None
    _landmark = None
    bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces>0:
       det = bounding_boxes[:,0:4]
       img_size = np.asarray(img.shape)[0:2]
       bindex = 0
       if nrof_faces>1:
          bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
          img_center = img_size / 2
          offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
          offset_dist_squared = np.sum(np.power(offsets,2.0),0)
          bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
       _bbox = bounding_boxes[bindex, 0:4]
       _landmark = points[:, bindex].reshape( (2,5) ).T
    warped, M = face_preprocess.preprocess(img, image_size=image_size, bbox=_bbox, landmark = _landmark)
    bgr = warped[...,:]
    return bgr, M

def re_align(img_small_adv, img_small, img_origin, M):
    perturb = img_small_adv.astype(float) - img_small.astype(float)
    img_origin = img_origin.astype(float)
    mask = np.ones_like(img_small)
    if M is None:
       origin_shape = img_origin.shape
       det1 = int(0)
       det2 = int(0)
       size1 = origin_shape[0] - 2 * det1
       size2 = origin_shape[1] - 2 * det2
       perturb = cv2.resize(perturb, (size2, size1))
       img_origin[det1:(det1+size1), det2:(det2+size2),:] += perturb
    else:
       perturb = cv2.warpAffine(src=perturb, M=M,dsize=(img_origin.shape[1], img_origin.shape[0]), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
       adv = cv2.warpAffine(src=img_small_adv, M=M,dsize=(img_origin.shape[1], img_origin.shape[0]), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)

       mask = cv2.warpAffine(src=mask, M=M,dsize=(img_origin.shape[1], img_origin.shape[0]), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
       img_origin = img_origin * (mask == 0) + adv * (mask > 0)
#      img_origin += perturb
    img_origin = np.clip(img_origin, 0, 255)
    return img_origin
