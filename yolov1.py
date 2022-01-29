import random
import time
import sys
import json
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import cv2 as cv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.experimental.set_device_policy()
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


class YOLO(keras.models.Model):
    # Image martix format: (height, width, channel)  # For the sake of reading by rows.
    I = (192, 256, 3)  # Image size # Attention to the text above
    S = (6, 8)  # Windows amount (row, column)
    C = 4  # classes amount
    A = int(I[0] / S[0])  # the vertical length of a window
    B = int(I[1] / S[1])  # the horizental length of a window
    L = 5 + C
    lr = 1e-6  # Learning rate

    def LoadImage(self, filepath) -> np.ndarray:
        img = keras.preprocessing.image.load_img(filepath)
        return keras.preprocessing.image.img_to_array(img, dtype=np.float32)/255

    def LoadDataset(self, root):
        dataset_x = []
        dataset_y = []
        files = os.listdir(os.path.join(root, 'data'))
        count = 0
        for file in files:
            with open(os.path.join(root, 'data', file)) as f:
                data = json.load(f)
                img_file = data['img']
                c = data['class']
                x = data['x']
                y = data['y']
                w = data['w']
                h = data['h']
                img = self.LoadImage(os.path.join(root, 'res', img_file))
                dataset_x.append(img)
                Y = self.ComputeY(c, x, y, w, h)
                dataset_y.append(Y)
                count = count + 1
        with open(os.path.join(root, 'config.json')) as f:
            config = json.load(f)
            label = config['label']
        return tf.convert_to_tensor(dataset_x), tf.convert_to_tensor(dataset_y), label

    def ComputeY(self, c, x, y, w, h):
        # Y = (S,S,5*W+C)
        # 5 *W + C = [hasObj, x, y, w, h, ..., c, c, ...]
        Y = np.zeros((self.S[0], self.S[1], self.L), dtype=np.float32)
        for i in range(len(x)):
            # Find out which grid cell it is.
            midpoint_pos = (
                int(np.floor(y[i]/self.A)),
                int(np.floor(x[i]/self.B))
            )
            relative_mid = (
                (x[i]-midpoint_pos[1]*self.B)/self.A,
                (y[i]-midpoint_pos[0]*self.A)/self.B
            )
            relative_wh = (w[i] / self.B, h[i] / self.A)
            Y[midpoint_pos[0]][midpoint_pos[1]][0] = 1
            Y[midpoint_pos[0]][midpoint_pos[1]][1] = relative_mid[0]  # x(left)
            Y[midpoint_pos[0]][midpoint_pos[1]][2] = relative_mid[1]  # y(top)
            Y[midpoint_pos[0]][midpoint_pos[1]][3] = relative_wh[0]  # width
            Y[midpoint_pos[0]][midpoint_pos[1]][4] = relative_wh[1]  # height
            Y[midpoint_pos[0]][midpoint_pos[1]][5+c[i]] = 1  # class
        return Y

    def __init__(self):
        super().__init__()
        self.layer_conv1_conv = keras.layers.Conv2D(256, (4, 4), strides=(4, 4))
        self.layer_conv1_actv = keras.layers.LeakyReLU(alpha=0.2)

        self.layer_conv2_conv = keras.layers.Conv2D(256, (4, 4), strides=(4, 4))
        self.layer_conv2_actv = keras.layers.LeakyReLU(alpha=0.2)

        self.layer_conv3_conv = keras.layers.Conv2D(384, (2, 2), strides=(2, 2))
        self.layer_conv3_actv = keras.layers.LeakyReLU(alpha=0.2)

        self.layer_flatten = keras.layers.Flatten()
        self.layer_dense1_dense = keras.layers.Dense(512)
        self.layer_dense1_actv = keras.layers.LeakyReLU(alpha=0.2)

        self.layer_dense2_dense = keras.layers.Dense(512)
        self.layer_dense2_actv = keras.layers.LeakyReLU(alpha=0.2)
        self.layer_output = keras.layers.Dense(self.S[0] * self.S[1] * (self.L))

    def call(self, input):
        x = self.layer_conv1_conv(input)
        x = self.layer_conv1_actv(x)

        x = self.layer_conv2_conv(x)
        x = self.layer_conv2_actv(x)

        x = self.layer_conv3_conv(x)
        x = self.layer_conv3_actv(x)

        x = self.layer_flatten(x)
        x = self.layer_dense1_dense(x)
        x = self.layer_dense1_actv(x)
        x = self.layer_dense2_dense(x)
        x = self.layer_dense2_actv(x)
        output = self.layer_output(x)
        return tf.reshape(output, (output.shape[0], self.S[0], self.S[1], self.L))

    def computeIoU(self, y_true_xy, y_true_wh, y_pred_xy, y_pred_wh):
        point_leftTop = tf.stack((y_true_xy, y_pred_xy), -1)
        max_leftTop_xy = tf.reduce_max(point_leftTop[:, :, :], 2)

        point_rightBottom = tf.stack((y_true_xy + y_true_wh, y_pred_xy + y_pred_wh), -1)
        min_rightBottom_xy = tf.reduce_min(point_rightBottom[:, :, :], 2)

        wh = tf.abs(min_rightBottom_xy - max_leftTop_xy)

        S_common = wh[:, 0] * wh[:, 1]
        S1 = y_true_wh[:, 0] * y_true_wh[:, 1]
        S2 = y_pred_wh[:, 0] * y_pred_wh[:, 1]
        S_added = S1 + S2 - S_common
        return S_common / S_added

    def lossFunction(self, y_true, y_pred):
        # Create mask
        obj_mask = y_true[:, :, :, 0:1] > 0
        noobj_mask = y_true[:, :, :, 0:1] == 0
        obj_mask = tf.repeat(obj_mask, y_true.shape[3], 3)
        noobj_mask = tf.repeat(noobj_mask, y_true.shape[3], 3)

        # No obj Loss:
        y_true_noobj_filtered = tf.reshape(y_true[noobj_mask], (-1, self.L))
        y_pred_noobj_filtered = tf.reshape(y_pred[noobj_mask], (-1, self.L))
        loss_noobj = tf.square(y_true_noobj_filtered[:, 0] - y_pred_noobj_filtered[:, 0])

        # Obj Loss:
        y_true_obj_filtered = tf.reshape(y_true[obj_mask], (-1, self.L))
        y_pred_obj_filtered = tf.reshape(y_pred[obj_mask], (-1, self.L))

        # # To compute IoU # IoU is just a metric index, not a Loss.
        # y_true_wh = y_true_obj_filtered[:, 3:5]
        # y_true_point_left_top = (y_true_obj_filtered[:, 1:3] - 0.5 * y_true_wh)
        # y_pred_wh = y_pred_obj_filtered[:, 3:5]
        # y_pred_point_left_top = (y_pred_obj_filtered[:, 1:3] - 0.5 * y_pred_wh)
        # iou = self.computeIoU(y_true_point_left_top, y_true_wh, y_pred_point_left_top, y_pred_wh)
        # # To compute Confidence
        # class_mask = y_true_obj_filtered[:, 5:] > 0
        # confidence = iou * y_pred_obj_filtered[:, 5:][class_mask]
        # print(tf.reduce_mean(confidence))

        loss_confidence = tf.square(y_true_obj_filtered[:, 0] - y_pred_obj_filtered[:, 0])
        loss_xy = tf.square(y_true_obj_filtered[:, 1:3] - y_pred_obj_filtered[:, 1:3])
        loss_wh = tf.square(tf.sqrt(y_true_obj_filtered[:, 3:5]) - tf.sqrt(y_pred_obj_filtered[:, 3:5]))
        loss_class = tf.square(y_true_obj_filtered[:, 5:] - y_pred_obj_filtered[:, 5:])

        loss_confidence = tf.reduce_mean(loss_confidence)
        loss_noobj = tf.reduce_mean(tf.reduce_sum(loss_noobj))
        loss_xy = tf.reduce_mean(tf.reduce_sum(loss_xy, axis=[1]))
        loss_wh = tf.reduce_mean(tf.reduce_sum(loss_wh, axis=[1]))
        loss_class = tf.reduce_mean(tf.reduce_sum(loss_class, axis=[1]))
        return loss_confidence + loss_noobj + loss_xy + loss_wh + loss_class

    def trainBatch(self, x, y_true):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.call(x)
            loss = self.lossFunction(y_true, y_pred)
        gradients = tape.gradient(loss, self.variables)
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        optimizer.apply_gradients(zip(gradients, self.variables))
        return loss.numpy()

    def train(self, x: tf.Tensor, y: tf.Tensor, batch_size=16, epoch=10, shuffle=True, learning_rate=1e-6):
        self.lr = learning_rate
        for i in range(epoch):
            index = list(range(x.shape[0]))
            if shuffle:
                random.shuffle(index)
            print(f'Epoch: {i+1}/{epoch}')
            progress = [' ']*50
            # compute batches amount
            batch_len = int(np.ceil(x.shape[0] / batch_size))
            for j in range(batch_len):
                # Create a batch
                batch_x = list()
                batch_y = list()
                for m in index[batch_size*j:batch_size*(j+1)]:
                    batch_x.append(x[m])
                    batch_y.append(y[m])
                loss = self.trainBatch(tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y))
                # Print progress
                cur = int((j+1)*50/batch_len)
                progress[0:cur] = ['=']*cur
                sys.stdout.flush()
                sys.stdout.write(f"[{j+1}/{batch_len}] [{''.join(progress)}] Loss:{loss}")
                if j != batch_len - 1:
                    sys.stdout.write('\r')
                else:
                    sys.stdout.write('\n')

    def predictBatch(self, x):
        return self.call(x)

    def evaluateShow(self, test_x, test_y, label):
        positiveVal = 0.6
        imgScaleFact = 4
        cv.namedWindow('main')
        batch_size = test_x.shape[0]
        y_pred = self.predictBatch(test_x)
        for i in range(batch_size):
            img_bgr = cv.cvtColor(test_x[i], cv.COLOR_RGB2BGR)
            img_bgr = cv.resize(img_bgr, None, None, imgScaleFact, imgScaleFact)
            for r in range(self.S[0]):
                for c in range(self.S[1]):
                    if y_pred[i][r][c][0] > positiveVal:
                        x = y_pred[i][r][c][1]
                        y = y_pred[i][r][c][2]
                        w = y_pred[i][r][c][3]
                        h = y_pred[i][r][c][4]
                        x_ = test_y[i][r][c][1]
                        y_ = test_y[i][r][c][2]
                        w_ = test_y[i][r][c][3]
                        h_ = test_y[i][r][c][4]
                        class_name = label[np.argmax(y_pred[i, r, c, 5:])]
                        iou = self.computeIoU(
                            np.array([[x_ - 0.5*w_, y_ - 0.5*h_]]), np.array([[w_, h_]]),
                            np.array([[x - 0.5*w, y - 0.5*h]]), np.array([[w, h]])
                        ).numpy()[0]
                        point_o = (c*self.B, r*self.A)
                        point_mid = (point_o[0]+x*self.B, point_o[1]+y*self.A)
                        point_1 = (int(point_mid[0]-(w*self.B/2)) * imgScaleFact, int(point_mid[1]-(h*self.A/2)) * imgScaleFact)
                        point_2 = (int(point_mid[0]+(w*self.B/2)) * imgScaleFact, int(point_mid[1]+(h*self.A/2)) * imgScaleFact)

                        point_mid_ = (point_o[0]+x_*self.B, point_o[1]+y_*self.A)
                        point_1_ = (int(point_mid_[0]-(w_*self.B/2)) * imgScaleFact, int(point_mid_[1]-(h_*self.A/2)) * imgScaleFact)
                        point_2_ = (int(point_mid_[0]+(w_*self.B/2)) * imgScaleFact, int(point_mid_[1]+(h_*self.A/2)) * imgScaleFact)
                        cv.rectangle(img_bgr, point_1, point_2, (0, 0, 255), 1)
                        cv.rectangle(img_bgr, point_1_, point_2_, (255, 125, 125), 1)
                        cv.putText(img_bgr, f'{class_name}:{str(iou)[0:5]}', point_1,
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5,
                                   color=(125, 125, 125),
                                   thickness=1)
            cv.imshow('main', img_bgr)
            cv.waitKey()

    def predictShow(self, input_x, label):
        positiveVal = 0.8
        imgScaleFact = 4
        cv.namedWindow('main')
        batch_size = input_x.shape[0]
        y_pred = self.predictBatch(input_x)
        for i in range(batch_size):
            img_bgr = cv.cvtColor(test_x[i], cv.COLOR_RGB2BGR)
            img_bgr = cv.resize(img_bgr, None, None, imgScaleFact, imgScaleFact)
            for r in range(self.S[0]):
                for c in range(self.S[1]):
                    if y_pred[i][r][c][0] > positiveVal:
                        x = y_pred[i][r][c][1]
                        y = y_pred[i][r][c][2]
                        w = y_pred[i][r][c][3]
                        h = y_pred[i][r][c][4]
                        class_name = label[np.argmax(y_pred[i, r, c, 5:])]
                        point_o = (c*self.B, r*self.A)
                        point_mid = (point_o[0]+x*self.B, point_o[1]+y*self.A)
                        point_1 = (int(point_mid[0]-(w*self.B/2)) * imgScaleFact, int(point_mid[1]-(h*self.A/2)) * imgScaleFact)
                        point_2 = (int(point_mid[0]+(w*self.B/2)) * imgScaleFact, int(point_mid[1]+(h*self.A/2)) * imgScaleFact)
                        cv.rectangle(img_bgr, point_1, point_2, (0, 0, 255), 1)
                        cv.putText(img_bgr, class_name, point_1,
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5,
                                   color=(125, 125, 125),
                                   thickness=1)
            cv.imshow('main', img_bgr)
            cv.waitKey()

    def predictOutput(self, input_x, label):
        positiveVal = 0.8
        imgScaleFact = 4
        batch_size = input_x.shape[0]
        y_pred = self.predictBatch(input_x)
        for i in range(batch_size):
            img_bgr = cv.cvtColor(input_x[i], cv.COLOR_RGB2BGR)
            img_bgr = cv.resize(img_bgr, None, None, imgScaleFact, imgScaleFact)
            for r in range(self.S[0]):
                for c in range(self.S[1]):
                    if y_pred[i][r][c][0] > positiveVal:
                        x = y_pred[i][r][c][1]
                        y = y_pred[i][r][c][2]
                        w = y_pred[i][r][c][3]
                        h = y_pred[i][r][c][4]
                        class_name = label[np.argmax(y_pred[i, r, c, 5:])]
                        point_o = (c*self.B, r*self.A)
                        point_mid = (point_o[0]+x*self.B, point_o[1]+y*self.A)
                        point_1 = (int(point_mid[0]-(w*self.B/2)) * imgScaleFact, int(point_mid[1]-(h*self.A/2)) * imgScaleFact)
                        point_2 = (int(point_mid[0]+(w*self.B/2)) * imgScaleFact, int(point_mid[1]+(h*self.A/2)) * imgScaleFact)
                        cv.rectangle(img_bgr, point_1, point_2, (0, 0, 255), 1)
                        cv.putText(img_bgr, class_name, point_1,
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5,
                                   color=(125, 125, 125),
                                   thickness=1)
            cv.imwrite(f'{time.localtime()[0]}-{time.localtime()[1]}-{time.localtime()[2]}-{i}.jpg', img_bgr*255)


yolo = YOLO()
train_x, train_y, train_label = yolo.LoadDataset('./dataset/train/round')
test_x, test_y, test_label = yolo.LoadDataset('./dataset/test/round')
yolo(train_x[0:1].numpy())
yolo.load_weights("roundWeights.h5")
# yolo.train(train_x, train_y, batch_size=4, epoch=100, shuffle=True, learning_rate=1e-6)
# yolo.save_weights("roundWeights.h5")
print('Train训练集表现')
yolo.evaluateShow(train_x.numpy(), train_y, train_label)
print('Test测试集表现')
yolo.evaluateShow(test_x.numpy(), test_y, test_label)
