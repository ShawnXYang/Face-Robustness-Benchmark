import sys
sys.path.insert(0, 'face-alignment')
import face_alignment as face_alignment
import os
from skimage import io
import numpy as np
from scipy.misc import imread, imsave
import json
import glob
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


total_preds = {}
for idx, img_path in enumerate(glob.glob(r"data/lfw-112x112/**/*[jp][pn]g", recursive=True)):
    preds = fa.get_landmarks_from_image(img_path)
    tokens = img_path.split('/')
    key = '_'.join(tokens[3:])
    total_preds[key] = preds.tolist()

with open("data/lfw_aligned_landmarks.json","w") as f:
    json.dump(total_preds,f)
    print("Done...")

total_preds = {}
for idx, img_path in enumerate(glob.glob(r"data/ytf-112x112/**/*[jp][pn]g", recursive=True)):
    preds = fa.get_landmarks_from_image(img_path)
    tokens = img_path.split('/')
    key = '_'.join(tokens[3:])
    total_preds[key] = preds.tolist()

with open("data/ytf_aligned_landmarks.json","w") as f:
    json.dump(total_preds,f)
    print("Done...")

total_preds = {}
for idx, img_path in enumerate(glob.glob(r"data/cfp-112x112/**/*[jp][pn]g", recursive=True)):
    preds = fa.get_landmarks_from_image(img_path)
    tokens = img_path.split('/')
    key = '_'.join(tokens[3:])
    try:
        total_preds[key] = preds.tolist()
    except Exception as e:
        pass

with open("data/cfp_aligned_landmarks.json","w") as f:
    json.dump(total_preds,f)
    print("Done...")