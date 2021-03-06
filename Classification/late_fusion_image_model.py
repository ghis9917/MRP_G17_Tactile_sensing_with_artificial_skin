# -*- coding: utf-8 -*-
"""Kopie von Kopie von image_model_v3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/118eXdeDDOCYHJGmlBtJHV9-fJDZdwSBR
"""

# !pip install -U efficientnet

# !echo $PWD

# from google.colab import drive

# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive
# %cd MRP_data_Julia
# %ls

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, NearestNDInterpolator
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import itertools
# import torch

def normalize(data, data_mean, data_std):
    return (data - data_mean) / data_std


def denormalize(data, data_mean, data_std):
    return (data * data_std) + data_mean


# https://stackoverflow.com/questions/39403183/python-opencv-sorting-contours/39445901
def sort_contours(cnts, method="top-to-bottom"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts


class Data:

    def __init__(self, mask_name, data_name, img_size, max_pressure):
        self.sensor_pos, self.mask, self.center = self.process_location_sensors(mask_name, img_size)
        df = pd.read_csv(data_name)
        self.info = df[['id', 'shape', 'pressure', 'velocity']].drop_duplicates()
        self.grouped = df.groupby(['id', 'dangeours/safe', 'press/tap', 'big/small', 'dynamic/static'])
        self.img_size = img_size
        self.max_pressure = max_pressure
        self.X, self.Y = np.meshgrid(
            np.linspace(0, self.mask.shape[0], num=self.mask.shape[0]),
            np.linspace(0, self.mask.shape[1], num=self.mask.shape[1]))

    def process_location_sensors(self, mask_name, img_size):
        img = cv2.imread(mask_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (img_size, img_size))

        contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # contours = sort_contours(contours)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] +
                                                    cv2.boundingRect(ctr)[1] * img.shape[1])

        print("no of detected sensors", len(contours))

        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        # Initialize empty list
        sensor_pos = []
        center = []

        # For each list of contour points...
        for i in range(len(contours)):
            # Create a mask image that contains the contour filled in
            cimg = np.zeros_like(img)
            cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

            # Access the image pixels and create a 1D numpy array then add to list
            pts = np.where(cimg == [255])
            coordinates = np.asarray(list(zip(pts[0], pts[1])))
            sensor_pos.append(pts)

            M = cv2.moments(contours[i])
            center.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        return sensor_pos, img_gray, center

    def preprocess_sequence(self, ids):
        labels = []
        training_frames = []

        for name, touch in self.grouped:
            label = name[1:5]

            if not name[0] in ids:
                continue

            # print(name, label)
            frames = []

            for row_index, row in touch.iterrows():

                # linear interpolation
                # inter_li = LinearNDInterpolator(self.center, row[6:-3], 0)
                # li_img = inter_li(self.X, self.Y)
                

                # nearest interpolation
                # inter_ni = NearestNDInterpolator(self.center, row[6:-3], 0)
                # cu_img = inter_ni(self.X, self.Y).astype(float)
                ni_img = np.zeros_like(self.mask)

                # cubic interpolation
                # inter_cu = CloughTocher2DInterpolator(self.center, row[6:-3], 0)
                # cu_img = inter_cu(self.X, self.Y)
          
                for i in range(len(self.sensor_pos)):
                    ni_img[self.sensor_pos[i][:2]] = row[f'S{i}']
                    # li_img[self.sensor_pos[i][:2]] = row[f'S{i}']
                    # cu_img[self.sensor_pos[i][:2]] = row[f'S{i}']
                
                # ni_img = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(ni_img)))).astype('uint8') / 255.0
                ni_img = (ni_img + 2)/ (self.max_pressure + 2)
                # li_img = (li_img + 2)/ (self.max_pressure + 2)
                # cu_img = (cu_img + 2)/ (self.max_pressure + 2)

                # frames.append([ni_img, li_img, cu_img])
                frames.append([ni_img, ni_img, ni_img])
                # TODO enable
                # frames.append([ni_img])
                # frames.append(ni_img)
            frames = np.asarray(frames)
            # norm = np.full((frames.shape[0], 3, self.img_size, self.img_size), 255.0) # self.max_pressure)
            # frames = frames / norm
            labels.append(label)
            training_frames.append(frames)

        training_frames = np.asarray(training_frames)
        # merged = list(itertools.chain(*labels_org))
        labels = np.asarray(labels)

        return training_frames, labels

    def prepare_data_batch(self, ids):
        training_frames, labels = self.preprocess_sequence(ids)
        training_frames_np = np.asarray(training_frames)
        training_frames_np = np.moveaxis(training_frames_np, 2, -1)
        # print(training_frames_np.shape)
        labels_np = np.asarray(labels).astype('float32')

        return training_frames_np, labels_np
    
    def prepare_info(self, ids):
        info_new = self.info[self.info['id'].isin(ids)]
        # print(ids, info_new)
        return info_new

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, list_IDs, data, test=False, batch_size=20, shuffle=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param data: link to data class
        :param test: train or test set generator
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.data = data
        self.test = test
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))  # int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        if self.test:
            return self._generate_data_test(self.list_IDs[index * self.batch_size:(index + 1) * self.batch_size])
        else:
            return self._generate_data(self.list_IDs[index * self.batch_size:(index + 1) * self.batch_size])

    def _generate_data(self, list_ID_temp):
        """Generates data containing batch_size images
        :param list_ID_temp: list of label ids to load
        :return: batch of images
        """

        X, y = self.data.prepare_data_batch(list_ID_temp)

        return X, y

    def _generate_data_test(self, list_ID_temp):
        """Generates data containing batch_size images
        :param list_ID_temp: list of label ids to load
        :return: batch of images
        """
        X, y = self._generate_data(list_ID_temp)
        z = self.data.prepare_info(list_ID_temp)

        return X, y, z

def matrix_confusion(y_test, y_pred, title):

    labels = ['big/small', 'dynamic/static', 'press/tap', 'dangerous/safe']
    # create confusion matrix
    matrix = multilabel_confusion_matrix(y_test, y_pred)
    for i in range(4):
        # print(matrix[i])
        sns.heatmap(matrix[i], square=True, annot=True, fmt='d', cbar=False)
        # set title and labels
        plt.title('Confusion Matrix_' + labels[i])
        plt.ylabel('true label')
        plt.xlabel('predicted label')
        # plt.savefig('Plots/Classification/confusion_matrix_' + title + '.png')
        plt.show()


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = list(range(len(loss)))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0, 0.5)
    plt.legend()
    # plt.savefig('loss_%s.png' % model)
    plt.show()
    # plt.close()

def false_gestures(y_test, y_pred, info):
    statistics = {0: {'class': 'big-small', 'obj info': []},
                      1: {'class': 'dynamic-static', 'obj info': []},
                      2: {'class': 'press-tap', 'obj info': []},
                      3: {'class': 'dangerous-safe', 'obj info': []}}
    info_encoder = {'circle_no_blur_small':0., 'string':1., 'hand_small':2.,
    'circle_blur_small':3., 'hand':4., 'hand2':5., 'band':6., 'hand1':7.,
    'circle blur':8., 'fist1':9., 'circle no blur':10.}

    print('len info', len(info), 'len pred', len(y_pred))
    # print([(i, len(v), type(v)) for i, v in enumerate(info)])
    for i, (pred, test) in enumerate(zip(y_pred, y_test)):
      uncorrect_class = np.where(pred != test)[0]
      # print(pred, test, uncorrect_class)
      for missclassified in uncorrect_class:
        idx = int(i / 50)
        element = info[idx].iloc[i % 50].values
        # print(element)
        element[-3] = info_encoder[element[-3].split('.')[0]]
        element = np.asarray(element[1:]).astype('float64')
        statistics[missclassified]['obj info'].append(torch.from_numpy(element))
    missclassified_obj(statistics, info_encoder)

def missclassified_obj(statistics, info_encoder, model='Model'):
    list_shapes = list(info_encoder.values())
    max_pressure = 1300
    discretization = 13
    max_velocity = 14
    for c in statistics:
        t = np.squeeze(torch.stack(statistics[c]['obj info']).numpy())
        # binarize shapes and show bars
        bin_obj_shape = np.bincount(t[:, 0].astype(int))
        if bin_obj_shape.size < len(list_shapes):
            bin_obj_shape = np.pad(bin_obj_shape, (0, len(list_shapes) - bin_obj_shape.size), 'constant', constant_values=0)
        plt.bar(list_shapes, bin_obj_shape)
        plt.xticks(list_shapes, list(info_encoder.keys()))
        plt.title(f'Misclassification for class {statistics[c]["class"]} compared to object shape')
        plt.savefig(f'images9/statistics_{statistics[c]["class"]}shape{model}.png')
        plt.close()
        # binarize pressure values and show bars
        bin_pressure = np.bincount(t[:, 1].astype(int))
        if bin_pressure.size < max_pressure:
            bin_pressure = np.pad(bin_pressure, (0, max_pressure - bin_pressure.size), 'constant', constant_values=0)
        # discretize bins
        bin_pressure = np.asarray([np.sum(chunk) for chunk in np.split(bin_pressure, discretization)])
        plt.bar(np.arange(discretization), bin_pressure)
        plt.xticks(np.arange(discretization), np.arange(0, max_pressure, max_pressure / discretization))
        plt.title(f'Misclassification for class {statistics[c]["class"]} compared to pressure')
        plt.savefig(f'images9/statistics_{statistics[c]["class"]}pressure{model}.png')
        plt.close()
        # binarize velocity and show bars
        bin_velocity = np.bincount((t[:, -1] * 10).astype(int))
        if bin_velocity.size < max_velocity:
            bin_velocity = np.pad(bin_velocity, (0, max_velocity - bin_velocity.size), 'constant', constant_values=0)
        plt.bar(np.arange(max_velocity), bin_velocity)
        plt.xticks(np.arange(max_velocity), np.arange(max_velocity) / 10)
        plt.title(f'Misclassification for class {statistics[c]["class"]} compared to velocity')
        plt.savefig(f'images9/statistics_{statistics[c]["class"]}velocity{model}.png')
        plt.close()


def create_model(model_name, img_size):
    if model_name == "efficientnet":
        tf.keras.utils.generic_utils = tf.keras.utils

        # EfficientNetB0 architecture
        base_model = EfficientNetB0(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False

        model = tf.keras.models.Sequential()
        model.add(base_model)
        model.add(tf.keras.layers.Flatten())

        inputs = tf.keras.Input(shape=(30, img_size, img_size, 3))
        conv_2d_layer = model
        timedis = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
        lstm = tf.keras.layers.LSTM(32, return_sequences=True)(timedis)
        lstm2 = tf.keras.layers.LSTM(32)(lstm)
        output = tf.keras.layers.Dense(4, activation='sigmoid')(lstm2)

    elif model_name == "simpleconv_pooling":
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16, 3, input_shape=(img_size, img_size, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))
        model.add(tf.keras.layers.SpatialDropout2D(0.2))
        model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
        model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))
        model.add(tf.keras.layers.SpatialDropout2D(0.2))
        model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())

        inputs = tf.keras.Input(shape=(30, img_size, img_size, 3))
        conv_2d_layer = model
        timedis = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
        attention = tf.keras.layers.Attention(use_scale=True)([timedis, timedis])
        pool = tf.math.multiply(attention, timedis)
        pool = tf.keras.layers.GlobalAveragePooling1D()(pool)
        # attention = tf.keras.layers.GlobalAveragePooling1D()(attention)
        # pool = tf.keras.layers.Concatenate()([pool, attention])
        dense = tf.keras.layers.Dense(256, activation='relu')(pool)
        dense = tf.keras.layers.Dropout(0.2)(dense)
        dense = tf.keras.layers.Dense(32, activation='relu')(dense)
        dense = tf.keras.layers.Dropout(0.2)(dense)
        output = tf.keras.layers.Dense(4, activation='sigmoid')(dense)
      
    elif model_name == "simpleconv_lstm":
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16, 3, input_shape=(img_size, img_size, 3), activation='relu'))
        model.add(tf.keras.layers.SpatialDropout2D(0.1))
        model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))
        model.add(tf.keras.layers.SpatialDropout2D(0.2))
        model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
        model.add(tf.keras.layers.SpatialDropout2D(0.1))
        model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))
        model.add(tf.keras.layers.SpatialDropout2D(0.2))
        model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())

        inputs = tf.keras.Input(shape=(30, img_size, img_size, 3))
        conv_2d_layer = model
        timedis = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
        attention = tf.keras.layers.Attention(use_scale=True)([timedis, timedis])
        pool = tf.math.multiply(attention, timedis)
        # lstm = tf.keras.layers.LSTM(128, return_sequences=True)(pool)
        # lstm = tf.keras.layers.Dropout(0.2)(lstm)
        lstm2 = tf.keras.layers.GRU(64)(pool)
        dense = tf.keras.layers.Dropout(0.2)(lstm2)
        output = tf.keras.layers.Dense(4, activation='sigmoid')(dense)
        output.shape

    model_final = tf.keras.Model(inputs=inputs, outputs=output)
    model_final.compile(tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6), loss='binary_crossentropy',
                        metrics=[tf.keras.metrics.BinaryAccuracy()])
    model_final.summary()
    return model_final


def run_training_gen(model_final, train, val):
    history = model_final.fit(train, validation_data=val, epochs=20, validation_steps=8)
    model_final.save("my_model_9")
    return history


def testing_gen(model_final, test):
    labels = []
    pred = []
    info = []
    for x_test, y_test, z in test:
        score = model_final.evaluate(x_test, y_test)
        labels.append(y_test)
        info.append(z)
        predicted = np.round(model_final.predict(x_test))
        pred.append(predicted)
    labels = np.asarray(list(itertools.chain(*labels)))
    pred = np.asarray(list(itertools.chain(*pred)))
    print(labels.shape, pred.shape)
    matrix_confusion(labels, pred, "Conv with GRU")
    false_gestures(labels, pred, info)


def run():
    img_size = 64
    batch_size = 50
    batches = 4038
    id = list(range(batches))
    random.shuffle(id)
    print("train", id[:int(batches*0.7)], "val", id[int(batches*0.7):int(batches*0.9)], "test", id[int(batches*0.9):])

    with tf.device('/device:GPU:0'):
      data = Data("sensor_map_fem3.png", "results_fem_18.csv", img_size=img_size, max_pressure=16.0)
      training_generator = DataGenerator(id[:int(batches*0.7)], data, test=False, batch_size= batch_size, shuffle=False)
      validation_generator = DataGenerator(id[int(batches*0.7):int(batches*0.9)], data, test=False, batch_size= batch_size, shuffle=False)
      test_generator = DataGenerator(id[int(batches*0.9):], data, test=True, batch_size= batch_size, shuffle=False)

      model_name = "simpleconv_lstm"
      model_final = create_model(model_name, img_size)
      history = run_training_gen(model_final, training_generator, validation_generator)
      visualize_loss(history, "Conv with GRU")
      results = testing_gen(model_final, test_generator)
      print(results)


run()