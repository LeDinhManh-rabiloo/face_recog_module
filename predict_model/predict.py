# Recognize letter

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from sklearn import preprocessing

# CONFIG = tf.compat.v1.ConfigProto()
# SESS = tf.compat.v1.Session(config=CONFIG)
# GRAPH = tf.compat.v1.get_default_graph()

NUMBER_CLASS = 66
# LST_LABEL = ['53@rabiloo-Vũ Văn Minh', '47@rabiloo-Nguyễn Như Ý', '35@rabiloo-Đỗ Tuấn Anh',
# '52@rabiloo-Nguyễn Đình Ngọc', '38@rabiloo-Nguyễn Văn Hoàng', '58@rabiloo-Đinh Công Tuấn Anh',
# '55@rabiloo-Nguyễn Bá Sơn', '28@rabiloo-Nguyễn Hữu Vinh', '34@rabiloo-Đinh Thị Ngọc Mai',
# '43@rabiloo-Trần Viết Xuân', '39@rabiloo-Phạm Duy Hiếu', '51@rabiloo-Nguyễn Việt Tiến',
# '46@rabiloo-Nguyễn Thị Thùy Dương', '30@rabiloo-Nguyễn Thị Dương', '27@rabiloo-Phạm Khổng Khương
# Duy', '45@rabiloo-Trần Xuân Nam', '29@rabiloo-Nguyễn Văn Thường', '33@rabiloo-Nguyễn Xuân Nam',
# '48@rabiloo-Nguyễn Đức Long', '32@rabiloo-Vương Nguyễn Hoàng Yến', '59@rabiloo-Vũ Thuỳ Linh',
# '54@rabiloo-Trần Khánh Duy', '56@rabiloo-Lương Minh Dương', '63@rabiloo-Phạm Văn Mạnh',
# '26@rabiloo-Lương Tiến Đồng', '42@rabiloo-Trần Hoàng Tùng', '36@rabiloo-Hoàng Văn Toàn',
# '24@rabiloo-Đặng Tuấn Dương', '41@rabiloo-Nguyễn Thị Thùy Trang', '50@rabiloo-Lương Văn Vỹ',
# '44@rabiloo-Bùi Nhật Anh', '49@rabiloo-Dương Văn Công', '23@rabiloo-Lê Quang Thành', '31@rabiloo-Trương
# Thùy Lan Hương', '62@rabiloo-Nguyễn Trường Giang', '60@rabiloo-Bùi Ngọc An', '37@rabiloo-Nguyễn Thu
# Uyên', '25@rabiloo-Lê Quang Sơn', '57@rabiloo-Nguyễn Vũ Thu Phương', '40@rabiloo-Nguyễn Bắc Hải',
# '61@rabiloo-Đỗ Ngọc Đức', '1141360137@Manh', '1141360101@Nghia', '1141360102@Linh', '1141360103@Quan',
# '1141360104@Thien', '1141360105@Nguyet', '1141360106@Bac', '1141360107@Huy', '1141360108@Muoi', '1141360109@Thom',
# '1141360110@Anh', '1141360111@Hue', '1141360112@Luong', '1141360113@Thuy', '1141360114@Phong', '1141360115@Dat',
# '1141360116@Van', '1141360117@Huong', '1141360118@Hoan', '1141360119@Loi', '1141360120@Tan']
LST_LABEL = ['1141360117@Huong', '34@rabiloo-Đinh Thị Ngọc Mai', '28@rabiloo-Nguyễn Hữu Vinh',
             '39@rabiloo-Phạm Duy Hiếu', '29@rabiloo-Nguyễn Văn Thường', '55@rabiloo-Nguyễn Bá Sơn',
             '30@rabiloo-Nguyễn Thị Dương', '58@rabiloo-Đinh Công Tuấn Anh', '1141360114@Phong',
             '43@rabiloo-Trần Viết Xuân', '61@rabiloo-Lê Văn Dũng', '45@rabiloo-Trần Xuân Nam',
             '46@rabiloo-Nguyễn Thị Thùy Dương', '35@rabiloo-Đỗ Tuấn Anh', '51@rabiloo-Nguyễn Việt Tiến',
             '52@rabiloo-Nguyễn Đình Ngọc', '38@rabiloo-Nguyễn Văn Hoàng', '27@rabiloo-Phạm Khổng Khương Duy',
             '53@rabiloo-Vũ Văn Minh', '1141360107@Huy', '1141360105@Nguyet', '1141360120@Tan', '1141360123@ThuTrang',
             '1141360118@Hoan', '26@rabiloo-Lương Tiến Đồng', '1141360112@Luong', '1141360116@Van',
             '33@rabiloo-Nguyễn Xuân Nam', '47@rabiloo-Nguyễn Như Ý', '48@rabiloo-Nguyễn Đức Long',
             '63@rabiloo-Phạm Văn Mạnh', '54@rabiloo-Trần Khánh Duy', '24@rabiloo-Đặng Tuấn Dương',
             '42@rabiloo-Trần Hoàng Tùng', '1141360110@Anh', '1141360109@Thom', '56@rabiloo-Lương Minh Dương',
             '32@rabiloo-Vương Nguyễn Hoàng Yến', '41@rabiloo-Nguyễn Thị Thùy Trang', '1141360122@UyenNguyen',
             '59@rabiloo-Vũ Thuỳ Linh', '50@rabiloo-Lương Văn Vỹ', '1141360113@Thuy', '44@rabiloo-Bùi Nhật Anh',
             '1141360111@Hue', '1141360106@Bac', '49@rabiloo-Dương Văn Công', '36@rabiloo-Hoàng Văn Toàn',
             '1141360102@Linh', '1141360137@manh', '1141360108@Muoi', '1141360121@LeTrang', '1141360103@Quan',
             '62@rabiloo-Nguyễn Trường Giang', '60@rabiloo-Bùi Ngọc An', '31@rabiloo-Trương Thùy Lan Hương',
             '23@rabiloo-Lê Quang Thành', '37@rabiloo-Nguyễn Thu Uyên', '1141360119@Loi', '1141360101@Nghia',
             '64@rabiloo-Đỗ Ngọc Đức', '57@rabiloo-Nguyễn Vũ Thu Phương', '1141360104@Thien',
             '40@rabiloo-Nguyễn Bắc Hải', '1141360115@Dat', '25@rabiloo-Lê Quang Sơn']
# print(len(LST_LABEL))
OPTIMIZERS = "SGD"


class Classifier:
    def __init__(self, input_dim, path_model):
        # set_session(SESS)
        global graph
        graph = tf.get_default_graph()
        model = load_model(path_model)
        self.model = model
        self.label = self.fit_label()
        self.input_dim = input_dim

    @staticmethod
    def fit_label():
        # use Label Encoding to decode later
        le = preprocessing.LabelEncoder()
        labels = list(LST_LABEL)
        le.fit(labels)
        return le
    def recognize_folder(self, features):
        conditions = []
        labels = []
        for feature in features:
            with graph.as_default():
                # set_session(SESS)
                item = self.model.predict(feature.reshape(1, 128))
            char = list(item[0])
            maxx = max(char)
            indexx = char.index(maxx)
            if maxx >= 0.8:
                c = self.label.inverse_transform([indexx])
                conditions.append(round(maxx, 3))
                labels.append(str(c[0]))
            else:
                conditions.append(round(maxx, 3))
                labels.append('unknown')
        yield labels, conditions

