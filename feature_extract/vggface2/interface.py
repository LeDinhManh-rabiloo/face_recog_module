from os import listdir
from os.path import join
import PIL
import numpy as np
import torch
from PIL import Image
from imutils import paths
import random
import os

mean = (131.0912, 103.8827, 91.4953)
is_eval = True

# cal feature image
def cal_feature(model_eval, pil_img_array):
    pil_img_array = np.expand_dims(pil_img_array, axis=0)
    feature = model_eval(torch.Tensor(pil_img_array.transpose(0, 3, 1, 2).copy()))[
                  1].detach().cpu().numpy()[:,
              :, 0, 0]
    feature = feature / np.sqrt(np.sum(feature ** 2, -1, keepdims=True))
    return feature


# load image data
def load_data(path='', shape=None):
    short_size = 224.0
    crop_size = shape
    img = PIL.Image.open(path)
    im_shape = np.array(img.size)  # in the format of (width, height, *)
    img = img.convert('RGB')

    ratio = float(short_size) / np.min(im_shape)
    img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),  # width
                           int(np.ceil(im_shape[1] * ratio))),  # height
                     resample=PIL.Image.BILINEAR)

    x = np.array(img)  # image has been transposed into (height, width)
    x = cal_mean(x, crop_size)
    return x, img


# Call mean
def cal_mean(x, crop_size):
    newshape = x.shape[:2]
    h_start = (newshape[0] - crop_size[0]) // 2
    w_start = (newshape[1] - crop_size[1]) // 2
    x = x[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
    x = x - mean
    return x


class InterfaceVggface2:
    def __init__(self, option):
        if not hasattr(InterfaceVggface2, 'model_name') or option != self.model_name:
            try:
                from face_model import get_model
            except ImportError:
                from feature_extract.vggface2.face_model import get_model
            self.model_name = option
            print("InterfaceVggface2 using model " + option)
            self.interface_model = get_model(option, is_eval)

    def extract_file(self, path_image):
        img1_array, img1_resize = load_data(path_image, shape=(224, 224, 3))
        feature = cal_feature(self.interface_model, img1_array)
        return feature[0]

    def extract_folder(self, path_folder):
        lst_features = []
        lst_path_image = list(paths.list_images(path_folder))
        random.shuffle(lst_path_image)
        labels = [p.split(os.path.sep)[-2] for p in lst_path_image]
        for path_image in lst_path_image:
            lst_features.append(self.extract_file(path_image))
        return lst_features, labels, lst_path_image


# if __name__ == '__main__':
#     interface_obj = InterfaceVggface2("senet50_ft_pytorch")
#     fea = interface_obj.extract_file(
#         "/Users/duydq/rabiloo/research/face-recognition/data_face/tuan hung/tuan hung0_cut.png")
#     print(fea.shape)
