import os
import math
import logging

import pandas as pd
import numpy as np

import keras.backend as K


# Formats Position
format_position = lambda price: ("-$" if price < 0 else "+$") + "{0:.2f}".format(
    abs(price)
)


# Formats Currency
format_currency = lambda price: "${0:.2f}".format(abs(price))


def sigmoid(x):
    """Performs sigmoid operation"""
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t"""
    d = t - n_days + 1
    block = (
        data[d : t + 1] if d >= 0 else -d * [data[0]] + data[0 : t + 1]
    )  # pad with t0
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


def get_npast_data(data, t, n_days):
    """Returns an n-day state representation ending at time t"""
    d = t - n_days + 1
    if d >= 0:
        block = data[d : t + 1]
    else:
        padding = np.array([data[0] for _ in range(-d)])
        block = np.concatenate((padding, data[0 : t + 1]))

    return np.array([block])


def get_norm_candle(data, t):
    if t == 0:
        t = 1
    return [sigmoid(d) for d in (data[t] - data[t - 1])]


def show_train_result(result, val_position, initial_offset):
    """Displays training results"""
    if val_position == initial_offset or val_position == 0.0:
        logging.info(
            "Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}".format(
                result[0], result[1], format_position(result[2]), result[3]
            )
        )
    else:
        logging.info(
            "Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})".format(
                result[0],
                result[1],
                format_position(result[2]),
                format_position(val_position),
                result[3],
            )
        )


def show_eval_result(model_name, profit, initial_offset):
    """Displays eval results"""
    if profit == initial_offset or profit == 0.0:
        logging.info("{}: USELESS\n".format(model_name))
    else:
        logging.info("{}: {}\n".format(model_name, format_position(profit)))


def get_data(filename):
    """Reads stock data from csv file"""
    df = pd.read_csv(filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp")
    split_date = "25-Jun-2021"
    data = df.loc[df["timestamp"] > split_date].copy()
    return data.drop(columns="timestamp").to_numpy()


def get_data_small(filename):
    """Reads stock data from csv file"""
    df = pd.read_csv(filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp")
    return list(df["close"])


def switch_k_backend_device():
    """Switches `keras` backend from GPU to CPU if required.
    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"