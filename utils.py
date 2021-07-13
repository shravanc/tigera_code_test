import glob
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt



GENUINE_DOMAIN_PATH = '../datasets/genuine_domains/'
MALWARE_DOMAIN_PATH = '../datasets/malware_domains/'
domain_name_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, ':': 10,
                          '-': 11, '.': 12, '/': 13, '_': 14, 'a': 15, 'b': 16, 'c': 17, 'd': 18, 'e': 19, 'f': 20,
                          'g': 21, 'h': 22, 'i': 23, 'j': 24, 'k': 25, 'l': 26, 'm': 27, 'n': 28, 'o': 29, 'p': 30,
                          'q': 31, 'r': 32, 's': 33, 't': 34, 'u': 35, 'v': 36, 'w': 37, 'x': 38, 'y': 39, 'z': 40,
                          np.NaN: 41}


def domain_to_ints(domain):
    return [
        domain_name_dictionary.get(y, domain_name_dictionary.get(np.NaN))
        for y in domain.lower()
    ]


def prep_data(data, max_length):
    return sequence.pad_sequences(
        np.array([domain_to_ints(x) for x in data]), maxlen=max_length
    )


def prep_dataframe(data, max_length=255):
    X = (data["domain"]
         .apply(lambda x: domain_to_ints(x))
         .pipe(sequence.pad_sequences, maxlen=max_length))
    y = data["flag"]
    return X, y.to_numpy()


def get_and_split_data(max_length):
    df = pd.DataFrame()
    for name in glob.glob(GENUINE_DOMAIN_PATH + '*'):
        temp = pd.read_csv(name, names=['domain', 'flag'])
        temp['flag'] = 1
        df = pd.concat([df, temp])

    for name in glob.glob(MALWARE_DOMAIN_PATH + '*'):
        df = pd.concat([df, pd.read_csv(name, names=['domain', 'flag'])])

    t_df = df.sample(frac=0.85, random_state=1)
    v_df = df.drop(t_df.index)

    t_x, t_y = prep_dataframe(t_df, max_length)
    val_x, val_y = prep_dataframe(v_df, max_length)

    return t_x, val_x, t_y, val_y


def prepare_dataset(data, labels, batch=32, shuffle_buffer=50):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


def get_data(max_length, batch_size, shuffle_buffer=50):
    t_x, val_x, t_y, val_y = get_and_split_data(max_length)

    train_dataset = prepare_dataset(t_x, t_y, batch_size, shuffle_buffer)
    valid_dataset = prepare_dataset(val_x, val_y, batch_size, shuffle_buffer)

    return train_dataset, valid_dataset


def plot_curve(history):
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(len(mae))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, mae, label=['Training MAE'])
    plt.plot(epochs, val_mae, label=['Validation MAE'])
    plt.legend()
    plt.show()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, loss, label=['Training Loss'])
    plt.plot(epochs, val_loss, label=['Validation Loss'])
    plt.legend()
    plt.show()
