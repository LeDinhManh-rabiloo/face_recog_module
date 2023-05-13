from __future__ import print_function

from os.path import join, dirname

from torch.autograd import Variable
import cv2

from lightDS.data import *
from lightDS.light_face_ssd import build_ssd
import torch
import numpy as np
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
torch.set_default_tensor_type('torch.FloatTensor')


class LightDSFD:
    def __init__(self):
        model_path = join(dirname(__file__), "weights/light_DSFD.pth")
        cfg = widerface_640
        # WIDERFace_CLASSES = ['face']
        # num_classes = len(WIDERFace_CLASSES) + 1  # +1 background
        self.net = build_ssd('test', cfg['min_dim'], 2)  # initialize SSD
        # if torch.cuda.is_available():
        #     self.net.load_state_dict(torch.load(model_path))
        #     self.net.cuda()
        # else:
        self.net.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')))
        self.net.eval()
        print('Finished loading model!')

        # evaluation
        self.transform = TestBaseTransform((104, 117, 123))
        self.thresh = cfg['conf_thresh']

    def infer(self, img, shrink):
        if shrink != 1:
            img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0), volatile=True)
        # if torch.cuda.is_available():
        #     x = x.cuda()
        y = self.net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1] / shrink, img.shape[0] / shrink,
                              img.shape[1] / shrink, img.shape[0] / shrink])
        det = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= self.thresh:
                score = detections[0, i, j, 0]
                # label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                det.append([pt[0], pt[1], pt[2], pt[3], score])
                j += 1
        if not det:
            det = [[0.1, 0.1, 0.2, 0.2, 0.01]]
        det = np.array(det)

        keep_index = np.where(det[:, 4] >= 0)[0]
        det = det[keep_index, :]
        return det

    def vis_detections(self, im, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return im

        for i in inds:
            bbox = dets[i, :4]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
        return im

    def get_faces(self, im, dets, thresh=0.5):
        lst_faces = []
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return lst_faces

        for i in inds:
            bbox = dets[i, :4]
            lst_faces.append(im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
        return lst_faces

    def crop_face(self, frame, is_show=False):
        det = self.infer(frame, shrink=1)
        if is_show:
            im_face = self.vis_detections(frame, det, 0.6)
            return im_face
        lst_faces = self.get_faces(frame, det, 0.6)
        return lst_faces
