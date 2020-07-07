import os
import sys
from networks.MobileFace import MobileFace
from networks.Mobilenet import Mobilenet
from networks.ResNet import resnet
from networks.ShuffleNet import ShuffleNetV1
from networks.CosFace import CosFace
from networks.SphereFace import SphereFace
from networks.ArcFace import ArcFace
from networks.IR import IR


def getmodel(face_model):
    """
        select the face model according to its name
        :param face_model: string
        :param FLAGS: a tf FLAGS (should be replace later)
        :param is_use_crop: boolean, whether the network accepted cropped images or uncropped images
        :loss_type: string, the loss to generate adversarial examples
        return:
        a model class
    """
    img_shape = (112, 112)
    if face_model == 'MobileFace':
        model = MobileFace()
    elif face_model == 'Mobilenet':
        model = Mobilenet()
    elif face_model == 'ResNet50':
        model = resnet(depth=50)
    elif face_model == 'ShuffleNet_V1_GDConv':
        model = ShuffleNetV1(pooling='GDConv')
    elif face_model == 'CosFace':
        model = CosFace()
        img_shape = (112, 96)
    elif face_model == 'SphereFace':
        model = SphereFace()
        img_shape = (112, 96)
    elif face_model == 'ArcFace':
        model = ArcFace()
    else:
        raise Exception
    return model, img_shape
