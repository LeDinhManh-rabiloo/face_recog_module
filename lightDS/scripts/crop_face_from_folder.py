import random
import sys
from os import makedirs, listdir
from os.path import join, dirname, exists

import cv2
from imutils import paths
import os

# sys.path.append(sys.path[0] + "/../")

from lightDS.light_dsfd import LightDSFD

def resize_ratio_opencv(image):
    width = 600
    inter = cv2.INTER_AREA
    (h, w) = image.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))
    # resize the image
    image = cv2.resize(image, dim, interpolation=inter)
    return image


def cropface(pathImage='', pathOut=''):
    if not exists(pathOut):
        makedirs(pathOut)
    light_obj = LightDSFD()
    frame = cv2.imread(pathImage)
    frame = resize_ratio_opencv(frame)
    faces = light_obj.crop_face(frame)
    if len(faces) != 0:
        for face in faces:
            if face.size != 0:
                cv2.imwrite("{}/{}.jpg".format(pathOut, random.randint(1, 1000000)), face)
        return "HAVE FACE"
    else:
        return "No Face"
