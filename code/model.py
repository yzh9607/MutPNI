import os
import numpy as np
import tensorflow as tf
from Encoder import Encoder


def get_model():
    inputESM = tf.keras.layers.Input(shape=(201, 1280))
    inputProtT5 = tf.keras.layers.Input(shape=(201, 1024))
    inputEnergy = tf.keras.layers.Input(shape=(73))
    featureESM = tf.keras.layers.Dense(512)(inputESM)
    featureESM = tf.keras.layers.Dense(256)(featureESM)
    featureESM = Encoder(2, 256, 4, 1024, rate=0.3)(featureESM)
    featureESM = featureESM[:, 100, :]
    featureProtT5 = tf.keras.layers.Dense(512)(inputProtT5)
    featureProtT5 = tf.keras.layers.Dense(256)(featureProtT5)
    featureProtT5 = Encoder(2, 256, 4, 1024, rate=0.3)(featureProtT5)
    featureProtT5 = featureProtT5[:, 100, :]

    featureConcat = tf.keras.layers.Concatenate()([featureESM, featureProtT5, inputEnergy])
    feature = tf.keras.layers.Dense(512, input_dim=256, activation='relu')(featureConcat)
    feature = tf.keras.layers.Dense(256, activation='relu')(feature)
    feature = tf.keras.layers.Dense(128, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.1)(feature)
    y = tf.keras.layers.Dense(1)(feature)

    model = tf.keras.models.Model(inputs=[inputESM, inputProtT5, inputEnergy], outputs=y)
    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
    model.compile(loss='mean_squared_error', optimizer=adam)
    model.summary()
    return model

