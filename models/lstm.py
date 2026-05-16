import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def anorexia_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.Masking(mask_value=-1, input_shape=input_shape),
        keras.layers.LSTM(8),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    return model



def depression_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.Masking(mask_value=-1, input_shape=input_shape),
        keras.layers.LSTM(8),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    return model


def self_harm_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.Masking(mask_value=-1, input_shape=input_shape),
        keras.layers.LSTM(8),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    return model


