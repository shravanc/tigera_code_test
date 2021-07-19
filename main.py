from utils import get_data, plot_curve

import tensorflow as tf
from tensorflow.python.keras.callbacks import (EarlyStopping, ModelCheckpoint, TensorBoard)
from tensorflow.python.keras.layers import (Conv1D, MaxPooling1D, Embedding,)
from tensorflow.python.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Model

# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name='cpu')   # GPU:  243s  CPU: 80s

EPOCHS = 5
MAX_LENGTH = 100
BATCH_SIZE = 64
SHUFFLE_BUFFER = 50

MODEL_PATH = "/tmp/models/"


def build_model():

    inp = Input(shape=(MAX_LENGTH,), name="Input")
    embedding = Embedding(input_dim=128, output_dim=128, input_length=MAX_LENGTH, name="Embedding")(inp)

    conv1 = Conv1D(128, 3, padding="same", strides=1, name="Conv1")(embedding)
    max_pool1 = MaxPooling1D(pool_size=2, padding="same")(conv1)

    conv2 = Conv1D(128, 2, padding="same", strides=1, name="Conv2")(max_pool1)
    max_pool2 = MaxPooling1D(2, padding="same")(conv2)

    flatten = Flatten()(max_pool2)

    dense = Dense(64)(flatten)

    drop1 = Dropout(0.5)(dense)

    output = Dense(1, activation="sigmoid")(drop1)

    model = Model(inputs=inp, outputs=output)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["mae", "mean_squared_error", "acc"]
    )

    return model


def get_callbacks():
    early_stopping = EarlyStopping(monitor="val_mae", patience=5)
    # saver = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_mae")
    # grapher = TensorBoard(log_dir=GRAPH_LOG, write_graph=True)
    return [early_stopping] #, saver] #, grapher]


def run_training(model, train_dataset, valid_dataset):
    results = model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        validation_data=valid_dataset,
    )
    return results


def train(train_dataset, valid_dataset):
    model = build_model()
    history = run_training(model, train_dataset, valid_dataset)
    # plot_curve(history)
    tf.keras.models.save_model(model, MODEL_PATH)
    return history


def main():
    train_dataset, valid_dataset = get_data(
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle_buffer=SHUFFLE_BUFFER,
    )
    return train(
        train_dataset,
        valid_dataset,
    )


# main()

