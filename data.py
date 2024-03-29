import glob
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt


GENUINE_DOMAIN_PATH = './datasets/genuine_domains/'
MALWARE_DOMAIN_PATH = './datasets/malware_domains/'

# Dictionary for allowed characters in domain
domain_name_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, ':': 10,
                          '-': 11, '.': 12, '/': 13, '_': 14, 'a': 15, 'b': 16, 'c': 17, 'd': 18, 'e': 19, 'f': 20,
                          'g': 21, 'h': 22, 'i': 23, 'j': 24, 'k': 25, 'l': 26, 'm': 27, 'n': 28, 'o': 29, 'p': 30,
                          'q': 31, 'r': 32, 's': 33, 't': 34, 'u': 35, 'v': 36, 'w': 37, 'x': 38, 'y': 39, 'z': 40,
                          np.NaN: 41}


def domain_to_ints(domain):
    """
    example:
    input: domain="abc"
    return: [15, 16, 17]
    """
    return [
        domain_name_dictionary.get(y, domain_name_dictionary.get(np.NaN))
        for y in domain.lower()
    ]


def prep_data(data, max_length=10):
    """
    Used while predicting
    example:
    input: ["abc"], max_length=10
    return: [[ 0  0  0  0  0  0  0 15 16 17]]
    """
    return sequence.pad_sequences(
        np.array([domain_to_ints(x) for x in data]), maxlen=max_length
    )


def prep_dataframe(data, max_length=255):
    """
    Returns "X" converting each domain to numerical sequence and label numpy array
    """
    X = (data["domain"]
         .apply(lambda x: domain_to_ints(x))
         .pipe(sequence.pad_sequences, maxlen=max_length))
    y = data["flag"]
    return X, y.to_numpy()


def genuine_data():
    df = pd.DataFrame()
    for name in glob.glob(GENUINE_DOMAIN_PATH + '*'):
        temp = pd.read_csv(name, names=['domain', 'flag'])
        temp['flag'] = 1
        df = pd.concat([df, temp])

    return df


def malware_data():
    df = pd.DataFrame()
    for name in glob.glob(MALWARE_DOMAIN_PATH + '*'):
        temp = pd.read_csv(name, names=['domain', 'flag'])
        temp['flag'] = 0
        df = pd.concat([df, temp])

    return df


def split_data(df, max_length=255):
    t_df = df.sample(frac=0.85, random_state=1)
    v_df = df.drop(t_df.index)

    t_x, t_y = prep_dataframe(t_df, max_length)
    val_x, val_y = prep_dataframe(v_df, max_length)

    return t_x, val_x, t_y, val_y


def gather_data():
    """
    Loop over genuine domain file lists and malware domain file lists
    returns training set and validation set
    """
    genuine_df = genuine_data()
    malware_df = malware_data()

    df = pd.concat([genuine_df, malware_df])

    return df


def prepare_dataset(data, labels, batch=32, shuffle_buffer=50):
    """
    Converts numpy array into tf.data type(scalable used on GCP)
    """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


def get_data(max_length, batch_size, shuffle_buffer=50):
    df = gather_data()
    t_x, val_x, t_y, val_y = split_data(df, max_length)

    train_dataset = prepare_dataset(t_x, t_y, batch_size, shuffle_buffer)
    valid_dataset = prepare_dataset(val_x, val_y, batch_size, shuffle_buffer)

    return train_dataset, valid_dataset


def plot_curve_v2(history, metric):
    metrics = history.history[metric]
    val_metrics = history.history[f"val_{metric}"]
    epochs = range(len(metrics))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metrics, label=[f"Training {metric}"])
    plt.plot(epochs, val_metrics, label=[f"Validation {metric}"])
    plt.legend()



