import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import random
from TimeHistory import *
from models import *
import time

global loc_input3
global time_input3
global activity_input3

global homes
global coms
global radiuses
global activity_entropy
global travel_diversity
global labels

def pickle_input(interval=0.5, cellsize=100, classes=2):
    loc_input = pickle.load(open('./user/inputs/loc_input_100', 'rb'), encoding='bytes')
    time_input = pickle.load(open('./user/inputs/time_input', 'rb'), encoding='bytes')
    activity_input = pickle.load(open('./user/inputs/activity_input', 'rb'), encoding='bytes')

    # global homes
    # global coms
    global radiuses
    global activity_entropy
    global travel_diversity
    # homes = pickle.load(open('./user/inputs/home_' + str(cellsize), 'rb'), encoding='bytes')
    # coms = pickle.load(open('./user/inputs/com_' + str(cellsize), 'rb'), encoding='bytes')
    radiuses = pickle.load(open('./user/inputs/radius_of_gyration', 'rb'), encoding='bytes')
    activity_entropy = pickle.load(open('./user/inputs/activity_entropy', 'rb'), encoding='bytes')
    travel_diversity = pickle.load(open('./user/inputs/travel_diversity', 'rb'), encoding='bytes')
    activity_entropy = np.array(list(activity_entropy))
    travel_diversity = np.array(list(travel_diversity))

    global labels
    labels = pickle.load(open('./user/inputs/labels', 'rb'), encoding='bytes')
    for i in range(len(loc_input)):
        # for j in range(len(loc_input[i])):
        loc_input[i] = list(keras.preprocessing.sequence.pad_sequences(loc_input[i], padding='post', maxlen=32))
        loc_input[i] = [list(loc_input[i][j]) for j in range(len(loc_input[i]))]

        time_input[i] = list(keras.preprocessing.sequence.pad_sequences(time_input[i], padding='post', maxlen=32))
        time_input[i] = [list(time_input[i][j]) for j in range(len(time_input[i]))]

        activity_input[i] = list(keras.preprocessing.sequence.pad_sequences(activity_input[i], padding='post', maxlen=32))
        activity_input[i] = [list(activity_input[i][j]) for j in range(len(activity_input[i]))]

        radiuses[i] = int(radiuses[i] / 100) + 1
        activity_entropy[i] = int(activity_entropy[i]/interval)+1
        travel_diversity[i] = int(travel_diversity[i] / interval)+1

        max_l = max(labels)
        min_l = min(labels)
        d_l = (max_l - min_l) / classes
        if labels[i] == max_l:
            labels[i] = classes - 1
        else:
            labels[i] = int((labels[i] - min_l) / d_l)
        # if labels[i]<(10558 + 113224)/2:
        #     labels[i] = 0
        # else:
        #     labels[i] = 1
    # if classes>2:
    #     labels = tf.one_hot(labels, classes)
    # else:
    #     labels = np.array(labels).reshape([-1, 1])
    loc_input = np.array(loc_input)
    print(loc_input.max())
    time_input = np.array(time_input)
    activity_input = np.array(activity_input)
    # homes = np.array(homes).reshape([-1, 1])
    # coms = np.array(coms).reshape([-1, 1])
    radiuses = np.array(radiuses).reshape([-1, 1])
    # labels = np.array(labels).reshape([-1, 1])
    activity_entropy = np.array(activity_entropy).reshape([-1, 1])
    travel_diversity = np.array(travel_diversity).reshape([-1, 1])

    global loc_input3
    global time_input3
    global activity_input3
    loc_input3 = np.hsplit(loc_input, 7)
    time_input3 = np.hsplit(time_input, 7)
    activity_input3 = np.hsplit(activity_input, 7)

global x_train
global x_val
global y_train
global y_val
def return_list(l, indexs):
    ret = []
    for i in l:
        t = [i[j] for j in indexs]
        t = np.array(t)
        ret.append(t)
    return ret

def x():
    return 0.7
def get_train(train_ratio = 0.7):
    # global homes
    # global coms
    global radiuses
    global activity_entropy
    global travel_diversity
    global x_train
    global x_val
    global y_train
    global y_val
    global labels
    # label_0_index = []
    # label_1_index = []
    indexes = [i for i in range(len(labels))]
    random.shuffle(indexes, x)
    # for i in range(len(labels)):
    #     if labels[i][0]==0:
    #         label_0_index.append(i)
    #     else:
    #         label_1_index.append(i)

    # random.shuffle(label_0_index)
    # random.shuffle(label_1_index)
    # train_index = label_0_index[:int(train_ratio*len(label_0_index))]
    # train_index.extend(label_1_index[:int(train_ratio*len(label_1_index))])
    # test_index = label_0_index[int(train_ratio * len(label_0_index)):]
    # test_index.extend(label_1_index[int(train_ratio * len(label_1_index)):])
    train_index = indexes[:int(train_ratio * len(labels))]
    test_index = indexes[int(train_ratio * len(labels)):]

    x_train = []
    x_val = []
    for i in range(7):
        x_train.append(return_list([loc_input3[i]], train_index)[0].reshape([
            return_list([loc_input3[i]], train_index)[0].shape[0], return_list([loc_input3[i]], train_index)[0].shape[2]]))    #np.array([loc_input3[i][j] for j in train_index])
        x_val.append(return_list([loc_input3[i]], test_index)[0].reshape([
            return_list([loc_input3[i]], test_index)[0].shape[0], return_list([loc_input3[i]], test_index)[0].shape[2]]))
        x_train.append(return_list([time_input3[i]], train_index)[0].reshape(
            [return_list([time_input3[i]], train_index)[0].shape[0], return_list([time_input3[i]], train_index)[0].shape[2]]))
        x_val.append(return_list([time_input3[i]], test_index)[0].reshape(
            [return_list([time_input3[i]], test_index)[0].shape[0], return_list([time_input3[i]], test_index)[0].shape[2]]
        ))
        x_train.append(return_list([activity_input3[i]], train_index)[0].reshape(
            [return_list([activity_input3[i]], train_index)[0].shape[0], return_list([activity_input3[i]], train_index)[0].shape[2]]
        ))
        x_val.append(return_list([activity_input3[i]], test_index)[0].reshape(
            [return_list([activity_input3[i]], test_index)[0].shape[0], return_list([activity_input3[i]], test_index)[0].shape[2]]
        ))

    x_train.extend(return_list([radiuses, activity_entropy, travel_diversity], train_index))
    x_val.extend(return_list([radiuses, activity_entropy, travel_diversity], test_index))
    y_train = return_list([labels], train_index)[0]
    y_val = return_list([labels], test_index)[0]


def acc_top4(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=4)

def get_model(classes=2):
    loc_input = keras.Input(shape=(None,), name='loc_input_1')
    time_input = keras.Input(shape=(None,), name='time_input_1')
    activity_input = keras.Input(shape=(None,), name='activity_input_1')
    # region_embedding = keras.Input(shape=(None, 32), name='region_embedding')

    # submodel = get_submodel()
    loc_embedding = layers.Embedding(8877, 32, name='loc_embedding')
    time_embedding = layers.Embedding(49, 32, name='time_embedding')
    activity_embedding = layers.Embedding(12, 32, name='activity_embedding')

    lstm_base = layers.LSTM(64, name='LSTM_base')
    lstm_high = layers.LSTM(64, return_state=True, name='LSTM_high')

    state = get_lstm_1(loc_embedding, time_embedding, activity_embedding, lstm_base, lstm_high,
                       loc_input, time_input, activity_input)

    input = []
    input.append(loc_input)
    input.append(time_input)
    input.append(activity_input)

    # loc_input_last = keras.Input(shape=(None,), name='loc_input_1_last')
    # time_input_last = keras.Input(shape=(None,), name='time_input_1_last')
    # activity_input_last = keras.Input(shape=(None,), name='activity_input_1_last')
    # input.append(loc_input_last)
    # input.append(time_input_last)
    # input.append(activity_input_last)

    # there are 7 days in one week, lstm_2 ~ lstm_7
    for i in range(2,8):
        loc_input = keras.Input(shape=(None,), name='loc_input_'+str(i))
        time_input = keras.Input(shape=(None,), name='time_input_'+str(i))
        activity_input = keras.Input(shape=(None,), name='activity_input_'+str(i))
        # region_embedding = keras.Input(shape=(None, 32), name='region_embedding')

        input.append(loc_input)
        input.append(time_input)
        input.append(activity_input)

        state = get_lstm_i(loc_embedding, time_embedding, activity_embedding, lstm_base,
                           lstm_high, state, i, loc_input, time_input, activity_input)


    state = state[0]

    # static sub network
    # home_input = layers.Input(shape=(1,), name='home_input')
    # com_input = layers.Input(shape=(1,), name='com_input')
    radius_input = layers.Input(shape=(1,), name='radius_input')
    activity_entropy_input = layers.Input(shape=(1,), name='activity_entropy_input')
    travel_diversity_input = layers.Input(shape=(1,), name='travel_diversity_input')
    input.extend([radius_input, activity_entropy_input, travel_diversity_input])

    pretrain_model = get_consistent_model(classes)

    # home_embedding = pretrain_model.get_layer('home_embedding')(home_input)
    # com_embedding = pretrain_model.get_layer('com_embedding')(com_input)
    # home_embedding.trainable = False
    # com_embedding.trainable = False
    radius_embedding = pretrain_model.get_layer('radius_embedding')
    # radius_embedding.trainable = False

    entropy_embedding = pretrain_model.get_layer('entropy_embedding')
    # entropy_embedding.trainable = False

    radius_embedding = radius_embedding(radius_input)
    activity_entropy_embedding = entropy_embedding(activity_entropy_input)
    travel_diversity_embedding = entropy_embedding(travel_diversity_input)

    # reshape_home = layers.Reshape((-1,), name='reshape_home')(home_embedding)
    # reshape_com = layers.Reshape((-1,), name='reshape_com')(com_embedding)
    reshape_activity_entropy = layers.Reshape((-1,), name='reshape_activity_entropy')(activity_entropy_embedding)
    reshape_travel_diversity = layers.Reshape((-1,), name='reshape_travel_diversity')(travel_diversity_embedding)
    reshape_radius = layers.Reshape((-1,), name='reshape_radius')(radius_embedding)

    # concatenate
    lstm_hidden = layers.Dense(32, activation='relu', name='Lstm_hidden_layer')(state)
    hidden1 = layers.concatenate([lstm_hidden, reshape_radius, reshape_activity_entropy, reshape_travel_diversity])
    hidden2 = layers.Dense(32, activation='relu', name='Hidden_layer')(hidden1)

    if classes == 2:
        pred = layers.Dense(1, activation='sigmoid', name='Prediction')(hidden2)
    else:
        pred = layers.Dense(classes, activation='softmax', name='Prediction')(hidden2)


    model = keras.Model(input, pred)
    return model

def make_or_restore_model(classes=2):
    checkpoint_dir = "./ckpt_"+str(classes)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        # latest_checkpoint = './ckpt_2_0/ckpt-loss=0.02'
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_model(classes)

def do_clustering(x_test, classes):
    # global x_val
    model = make_or_restore_model(classes)
    clustermodel = keras.Model(model.input,
                               model.get_layer('Hidden_layer').output)
    return clustermodel.predict(x_test)

pickle_input(classes=2)
get_train(train_ratio=0.7)
a = do_clustering(x_val, 3)
Kmeans(a, y_val.reshape([-1,]), classes=2)