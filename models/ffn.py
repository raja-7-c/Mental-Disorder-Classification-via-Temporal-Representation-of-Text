import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def anorexia_ffn(num_features):
    model = keras.Sequential([
        layers.Dense(20, activation='relu', input_shape=(num_features,)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(14, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(8, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(4, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

def depression_ffn(num_features):
    model = keras.Sequential([
        layers.Dense(20, activation='relu',input_shape=(num_features,)),
        layers.BatchNormalization(),

        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),

        layers.Dense(8, activation='relu'),
        layers.BatchNormalization(),

        layers.Dense(1, activation='sigmoid')
    ])

    return model

def self_harm_ffn(num_features):
    model = keras.Sequential([
        layers.Dense(20, activation='relu',input_shape=(num_features,)),
        layers.BatchNormalization(),

        layers.Dense(10, activation='relu'),
        layers.BatchNormalization(),

        layers.Dense(1, activation='sigmoid')
    ])

    return model