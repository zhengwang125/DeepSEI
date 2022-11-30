import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import os
import pickle

def acc_top4(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=4)

def get_model():
    input = layers.Input(shape=(None,), name='input')
    embed = layers.Embedding(83, 32, name='embedding', input_length=1)(input)

    output = layers.Dense(82, activation='softmax', name='output')(embed)
    output = layers.Reshape((output.shape[2],))(output)
    # tf.reshape(output, shape=[output.shape[0], output.shape[2]])
    model = keras.Model(input, output)
    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=[acc_top4]) #'categorical_accuracy'
    return model

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    # checkpoint_dir = "./ckpt_2_0"
    checkpoint_path = "../KDD2022/spatial_pre/cp-{epoch:04d}.ckpt_2_0"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # if checkpoints:
    #     latest_checkpoint = max(checkpoints, key=os.path.getctime)
    #     print("Restoring from", latest_checkpoint)
    #     model = get_model()
    #     model.load_weights(latest_checkpoint)
    #     return model
        # return keras.models.load_model(latest_checkpoint, custom_objects={"top_k_categorical_accuracy": acc_top4})
    if latest:
        print("Restoring from", latest)
        model = get_model()
        model.load_weights(latest)
        return model
    print("Creating a new model")
    return get_model()

def train(X, Y):
    model = make_or_restore_model()
    log_dir = "logs2/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # checkpoint_dir = "./ckpt_2_0"
    checkpoint_path = "../KDD2022/spatial_pre/cp-loss{loss:.2f}.ckpt_2_0"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    history = model.fit(x=X,
                        y=Y,
                        epochs=500,
                        # validation_data=(x_test, y_test),
                        shuffle=True,
                        batch_size=64,
                        callbacks=[tensorboard_callback,
                                   keras.callbacks.ModelCheckpoint( # checkpoint_dir + "/ckpt_2_0-loss={loss:.2f}"
                                       filepath=checkpoint_path,
                                       save_freq=37,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       monitor='loss',
                                       mode='min'
                                   )]
                        )

def training():
    seq = np.arange(1, 83)
    skipgram, _ = keras.preprocessing.sequence.skipgrams(seq, 82, 2, 0)
    X = []
    Y = []
    for x, y in skipgram:
        X.append(x-1)
        Y.append(y-1)

    Y = tf.one_hot(Y, 82)
    # Y = tf.reshape(Y, shape=[Y.shape[0], 1, Y.shape[1]])
    X = np.array(X)
    # X = X.reshape([X.shape[0],1])
    train(X, Y)

def output_emb():
    model = make_or_restore_model()
    submodel = keras.Model(model.input, model.get_layer('embedding').output)
    # model.get_layer('embedding').trainable = False
    # model.get_layer('embedding').name
    x = np.arange(0, 82)
    emb = submodel.predict(x)
    emb = emb.reshape([emb.shape[0], emb.shape[2]])
    print(emb)
    print(emb.shape)
    print(type(emb))
    pickle.dump(emb, open('region_vector', 'wb'), protocol=2 )

training()
# output_emb()
