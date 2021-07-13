import tensorflow as tf
from utils import prep_data
from main import MAX_LENGTH, MODEL_PATH


def predict(sample_domain):
    inp = prep_data([sample_domain], MAX_LENGTH)
    print(inp)
    model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict(inp)
    print(prediction)


sample_domain = "google.com" # Predicted value: 0.9999827 (Valid)
sample_domain = "kwtoestnessbiophysicalohax.com"  # Predicted value: 0.00258332 (DGA)
predict(sample_domain)
