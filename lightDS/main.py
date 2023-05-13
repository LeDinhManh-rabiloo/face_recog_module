import io
from os import getenv

import cv2
import numpy as np
import zmq
from PIL import Image
from dotenv import load_dotenv

from lightDS.light_dsfd import LightDSFD

# define setup for receive message
load_dotenv()
context = zmq.Context()
receiver = context.socket(zmq.SUB)
receiver.connect(getenv("HOST_ZMQ"))  # 0.0.0.0
receiver.setsockopt_string(zmq.SUBSCRIBE, "")

# define setup for send message
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:9004")
dsfd_obj = LightDSFD()


def receive_message():
    try:
        print("System ready for receive data ... ")
        while True:
            #  Receiving image in bytes
            image_bytes = receiver.recv()
            print("receive image")
            #  Decoding the image -- Python's PIL.Image library is used for decoding

            image = np.array(Image.open(io.BytesIO(image_bytes)))
            # image = Image.open(io.BytesIO(image_bytes))
            faces = dsfd_obj.crop_face(image)

            yield faces
    except Exception as ex:
        print(ex)


def send_to_recognition():
    for faces in receive_message():
        print(len(faces))
        for face in faces:
            print("SEND FACE ++++++")
            retval, buf = cv2.imencode(".jpg", face)
            socket.send(buf)


if __name__ == '__main__':
    send_to_recognition()
