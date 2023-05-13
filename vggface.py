from feature_extract.ef import ExtractFeature
from predict_model.predict import Classifier
import numpy as np
from os.path import join, dirname

extract_feature_dict = {
    "senet50_128_pytorch": "vggface2",
}

feature_obj = ExtractFeature("senet50_128_pytorch")


def get_many_feature(path_folder):
    lst_feature = feature_obj.get_feature_folder(path_folder)
    return lst_feature


def query_model_folder(feature):
    classifier = Classifier(128, join(dirname(__file__), "predict_model/models/SGD/senet50_128_eval.h5"))
    return classifier.recognize_folder(feature)


def check_manny(path_folder):
    fea, labels, path_img = get_many_feature(path_folder)
    result = []
    for kq in query_model_folder(np.asarray(fea)):
        for i in range(len(labels)):
            result.append([kq[0][i], kq[1][i], path_img[i]])
    return result
