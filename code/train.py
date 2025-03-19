import os
import numpy as np
import joblib
import tensorflow as tf
from model import get_model


def train():
    X_train_esm = np.load('../data/train_esm.npy')
    X_train_protT5 = np.load('../data/train_protT5.npy')
    X_train_energy = np.load('../data/train_energy.npy')
    y_train = np.load('../data/train_ddg.npy')

    batch_size = 256
    earlyStopPatience = 10
    monitor = 'val_loss'
    input_train = [X_train_esm, X_train_protT5, X_train_energy]

    encoder_model = get_model()
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                     patience=earlyStopPatience,
                                                     verbose=1,
                                                     mode='auto')
    log_dir = "./model/logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    encoder_model.fit(x=input_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=10000,
                      verbose=1,
                      callbacks=[earlystopping],
                      validation_split=0.1,
                      shuffle=True)
    encoder_model.save('./model/encoder_model.h5')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    train()
