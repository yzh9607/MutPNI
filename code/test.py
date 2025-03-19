import os
import numpy as np
import tensorflow as tf
from model import get_model
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def test():
    X_test_esm = np.load('../data/test_esm.npy')
    X_test_protT5 = np.load('../data/test_protT5.npy')
    X_test_energy = np.load('../data/test_energy.npy')
    y_test = np.load('../data/test_ddg.npy')
    input_test = [X_test_esm, X_test_protT5, X_test_energy]

    encoder_model = load_model('../model/encoder_model.h5')
    y_pred = encoder_model.predict(input_test).reshape(-1, )
    pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print('Pearson Correlation Coefficient:', pearson_corr)
    print('Root Mean Square Error:', rmse)
    print('Mean Absolute Error:', mae)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    test()
