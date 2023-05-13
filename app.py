from flask import Flask, flash, render_template, url_for, Response, request, jsonify, session
from flask_cors import CORS
import base64
import os
from datetime import date, datetime
from lightDS.scripts import crop_face_from_folder
import vggface
from os import listdir, makedirs
from os.path import join, isdir, exists, dirname
import shutil

import json

app = Flask(__name__)
cors = CORS(app)
IS_EVAL = True
OPTIMIZERS = 'SGD'


@app.route('/api/v1/checkImage', methods=['POST'])
def checkImage():
    if request.method == 'POST':
        data_form = request.get_json()
        images = data_form.get('images')
        checkday = date.today()
        face_crop = ''
        resultDetesct = []
        for image in images:
            today = datetime.now()
            date_take = date.today().strftime("%d_%m_%Y")
            timeToday = today.strftime("%d.%m.%Y %H:%M:%S,%f")
            classId = data_form.get('classId')
            pathRoot = dirname(__file__)
            date_tiem = datetime.strptime(timeToday, "%d.%m.%Y %H:%M:%S,%f").strftime('%s.%f')
            imagepath = pathRoot + '/static/' + date_take + '/' + classId + "_" + date_take
            if not os.path.exists(imagepath):
                os.makedirs(imagepath)
            milisecond = int(float(date_tiem) * 1000)
            imageName = imagepath + '/' + classId + "@" + str(milisecond) + ".jpg"
            pathOut = pathRoot +"/static/face/" + classId + '@' + str(milisecond)
            with open(imageName, 'wb') as f:
                f.write(base64.b64decode(image.split('base64,')[1]))
            face_crop = crop_face_from_folder.cropface(imageName, pathOut)
            if face_crop == "HAVE FACE":
                ketqua = vggface.check_manny(pathOut)
                for result in ketqua:
                    if result[0] != 'unknow':
                        resultDetesct.append([result[0].split('@')[0], '/static' + result[2].split('/static')[1]])
                    else:
                        resultDetesct.append([result[0], '/static' + result[2].split('/static')[1]])
            else:
                return jsonify({'about': "No face"})
        return jsonify({'about': "success", 'data': resultDetesct})


@app.route('/api/v1/deleteImage', methods=['POST'])
def deleteImage():
    if request.method == 'POST':
        data_form = request.get_json()
        pathRoot = dirname(__file__)
        image = data_form.get('imageLink').split('localhost:5000/')[1]
        shutil.rmtree(image)
        return jsonify({'about': 'success'})


if __name__ == "__main__":
    app.run(debug=True)
