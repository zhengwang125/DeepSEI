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
import tensorflow_addons as tfa

global homes
global coms
global radiuses
global activity_entropy
global travel_diversity
global labels

def pickle_input(interval=0.5, cellsize=100, classes=2):
    global radiuses
    global activity_entropy
    global travel_diversity
    # homes = pickle.load(open('./user/inputs/home_' + str(cellsize), 'rb'), encoding='bytes')
    # coms = pickle.load(open('./user/inputs/com_' + str(cellsize), 'rb'), encoding='bytes')
    radiuses = pickle.load(open('./user/inputs/radius_of_gyration', 'rb'), encoding='bytes')
    activity_entropy = pickle.load(open('./user/inputs/activity_entropy', 'rb'), encoding='bytes')
    travel_diversity = pickle.load(open('./user/inputs/travel_diversity', 'rb'), encoding='bytes')
    radiuses = np.array(list(radiuses))
    activity_entropy = np.array(list(activity_entropy))
    travel_diversity = np.array(list(travel_diversity))

    global labels
    labels = pickle.load(open('./user/inputs/labels', 'rb'), encoding='bytes')
    for i in range(len(radiuses)):
        radiuses[i] = int(radiuses[i]/100)+1
        activity_entropy[i] = int(activity_entropy[i]/interval)+1
        travel_diversity[i] = int(travel_diversity[i] / interval)+1

        max_l = max(labels)
        min_l = min(labels)
        d_l = (max_l-min_l)/classes
        if labels[i] == max_l:
            labels[i] = classes-1
        else:
            labels[i] = int((labels[i]-min_l)/d_l)
        # if labels[i]<(10558 + 113224)/2:
        #     labels[i] = 0
        # else:
        #     labels[i] = 1
    if classes > 2:
        labels = tf.one_hot(labels, classes)
    else:
        labels = np.array(labels).reshape([-1, 1])


    # homes = np.array(homes).reshape([-1, 1])
    # coms = np.array(coms).reshape([-1, 1])
    radiuses = np.array(radiuses).reshape([-1, 1])
    # labels = np.array(labels).reshape([-1, 1])
    activity_entropy = np.array(activity_entropy).reshape([-1, 1])
    travel_diversity = np.array(travel_diversity).reshape([-1, 1])


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
def get_train(train_ratio = 0.8):
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
    #
    # random.shuffle(label_0_index)
    # random.shuffle(label_1_index)
    train_index = indexes[:int(train_ratio*len(labels))]
    test_index = indexes[int(train_ratio*len(labels)):]
    # train_index = label_0_index[:int(train_ratio*len(label_0_index))]
    # train_index.extend(label_1_index[:int(train_ratio*len(label_1_index))])
    # test_index = label_0_index[int(train_ratio * len(label_0_index)):]
    # test_index.extend(label_1_index[int(train_ratio * len(label_1_index)):])

    x_train = return_list([radiuses, activity_entropy, travel_diversity], train_index)
    x_val = return_list([radiuses, activity_entropy, travel_diversity], test_index)
    y_train = return_list([labels], train_index)[0]
    y_val = return_list([labels], test_index)[0]
    # x_train, x_val, y_train, y_val = train_test_split(x_train, labels, test_size=0.3, random_state=1)

def acc_top4(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=4)

def get_entropy_model():
    input = layers.Input(shape=(1,), name='input')
    embed = layers.Embedding(13, 32, name='entropy_embedding', input_length=1)(input)

    output = layers.Dense(12, activation='softmax', name='output')(embed)
    output = layers.Reshape((output.shape[2],))(output)
    # tf.reshape(output, shape=[output.shape[0], output.shape[2]])
    model = keras.Model(input, output)
    model.summary()
    # model.compile(optimizer=tf.optimizers.Adam(1e-3), loss='categorical_crossentropy',
    #               metrics=[acc_top4])  # 'categorical_accuracy'
    checkpoint_path = "entropy_pre/cp-{epoch:04d}.ckpt_2_0"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    return model

def get_radius_model():
    input = layers.Input(shape=(1,), name='input')
    embed = layers.Embedding(83, 32, name='radius_embedding', input_length=1)(input)

    output = layers.Dense(82, activation='softmax', name='output')(embed)
    output = layers.Reshape((output.shape[2],))(output)
    # tf.reshape(output, shape=[output.shape[0], output.shape[2]])
    model = keras.Model(input, output)
    model.summary()
    # model.compile(optimizer=tf.optimizers.Adam(1e-3), loss='categorical_crossentropy',
    #               metrics=[acc_top4])  # 'categorical_accuracy'
    checkpoint_path = "radius_pre/cp-{epoch:04d}.ckpt_2_0"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    return model

def get_network(classes=2):
    # home_input = layers.Input(shape=(1,), name='home_input')
    # com_input = layers.Input(shape=(1,), name='com_input')
    radius_input = layers.Input(shape=(1,), name='radius_input')
    activity_entropy_input = layers.Input(shape=(1,), name='activity_entropy_input')
    travel_diversity_input = layers.Input(shape=(1,), name='travel_diversity_input')
    input = [radius_input, activity_entropy_input, travel_diversity_input]

    # home_embedding = layers.Embedding(143, 32, name='home_embedding')(home_input)
    # com_embedding = layers.Embedding(145, 32, name='com_embedding')(com_input)
    radius_model = get_radius_model()
    radius_embedding = radius_model.get_layer('radius_embedding')
    # radius_embedding.trainable = False
    radius_embedding = radius_embedding(radius_input)

    entropy_model = get_entropy_model()
    entropy_embedding = entropy_model.get_layer('entropy_embedding')
    # entropy_embedding.trainable = False

    activity_entropy_embedding = entropy_embedding(activity_entropy_input)
    travel_diversity_embedding = entropy_embedding(travel_diversity_input)

    # reshape_home = layers.Reshape((-1, ), name='reshape_home')(home_embedding)
    # reshape_com = layers.Reshape((-1,), name='reshape_com')(com_embedding)

    reshape_activity_entropy = layers.Reshape((-1,), name='reshape_activity_entropy')(activity_entropy_embedding)
    reshape_travel_diversity = layers.Reshape((-1,), name='reshape_travel_diversity')(travel_diversity_embedding)
    reshape_radius = layers.Reshape((-1,), name='reshape_radius')(radius_embedding)

    hidden1 = layers.concatenate([reshape_radius, reshape_activity_entropy, reshape_travel_diversity])
    hidden2 = layers.Dense(32, activation='relu', name='Hidden_layer')(hidden1)

    if classes == 2:
        pred = layers.Dense(1, activation='sigmoid', name='Prediction')(hidden2)
    else:
        pred = layers.Dense(classes, activation='softmax', name='Prediction')(hidden2)

    model = keras.Model(input, pred)
    return model
    # model.summary()

def make_or_restore_model(classes=2):
    checkpoint_dir = "./ckpt_static_"+str(classes)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        # latest_checkpoint = './ckpt_static_2/ckpt_2_0-loss=0.48'
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_network(classes)
def weighted_f1_tfaddon(y_true, y_pred, num_class=3):
    metric = tfa.metrics.F1Score(num_classes=num_class, average='weighted')
    if num_class>2:
        y_pred = tf.one_hot(tf.argmax(y_pred, 1), depth=num_class)
    metric.update_state(y_true, y_pred)
    result = metric.result()
    return result.numpy()

def get_f1(y_true, y_pred):
    m1 = keras.metrics.Precision()
    m2 = keras.metrics.Recall()
    m1.update_state(y_true, y_pred)
    m1.result().numpy()
    m2.update_state(y_true, y_pred)
    m2.result().numpy()
    return 2*m1*m2/(m1+m2)

def train(x_train, y_train, x_test, y_test, classes=2):
    model = make_or_restore_model(classes)
    model.summary()
    if classes == 2:
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss='binary_crossentropy',
                      metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy'), keras.metrics.Recall(name='recall'),
                               keras.metrics.Precision(name='Precision')]  # 评价指标有准确率和PRC-AUC
                      )
    else:
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss='categorical_crossentropy',
                      metrics=[keras.metrics.CategoricalAccuracy(name='accuracy'),
                               keras.metrics.Recall(name='recall'),
                               keras.metrics.Precision(name='Precision')]  # 评价指标有准确率和PRC-AUC
                      )
    log_dir = "logs_static_"+str(classes)+"/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_dir = "./ckpt_static_"+str(classes)
    # weight_for_0 = (1 / len(x_train)+len(x_test)) * (len(x_train)+len(x_test) / 2.0)
    # weight_for_1 = (1 / 33) * (len(x_train)+len(x_test) / 2.0)
    #
    # class_weight = {0: weight_for_0, 1: weight_for_1}

    time_callback = TimeHistory()
    history = model.fit(x=x_train,
              y=y_train,
              epochs=50,
                        batch_size=256,
                        shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback,
                         time_callback,
                         keras.callbacks.ModelCheckpoint(
                             filepath=checkpoint_dir + "/ckpt_2_0-loss={loss:.2f}", save_freq=100
                         )]
              # class_weight=class_weight
              )
    print('epochs:', time_callback.times)
    print('total:', time_callback.totaltime)
    print(history.history.keys())
    precision = history.history['Precision']
    recall = history.history['recall']

    print(precision)
    print(recall)
    f1 = [2*i*j/(i+j) for (i,j) in zip(precision, recall)]
    print(f1)

def evaluate(xtest, ytest):
    model = make_or_restore_model()
    ypre = model.predict(xtest)


pickle_input(classes=2)
get_train()
train(x_train, y_train, x_val, y_val, classes=2)