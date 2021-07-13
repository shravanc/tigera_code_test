# Python Version:
3.8.11

# TensorFlow Version:
2.4.0

# Configuration:

## main.py
```
EPOCHS = 100
MAX_LENGTH = 100
BATCH_SIZE = 64
SHUFFLE_BUFFER = 50
MODEL_PATH = "/tmp/models/"
```

## utils.py
```
GENUINE_DOMAIN_PATH = '../datasets/genuine_domains/'
MALWARE_DOMAIN_PATH = '../datasets/malware_domains/'
```

# Train:
```
python main.py
```

# Prediction:
```
python predict.py
```

# Malware family considered:
```
banjori
chinad
corebot
fobber
```

# Platfor Develpoed:
```
MacOS: BigSur
M1 Chip
```

# Solution Delivery:
## Document the approach, pros and cons with the selected approach.
 - approach: 
    - pre_processing: Convert the text data to numericals using "domain_name_dictionary" -> padding is applied to maintain constant length(Since NLP problem) -> split data into 85% traininig data and 15% validation data.
    - build_model: Since binary classification "sigmoid" activation function is used and "binary_crossentropy" loss function is used, with early_stopping callbacks to avoid unecessary training iterations.
    - training: training is carried on the tf.data.Dataset type with shuffling and prefetch to fetch the dataset before next training iteration and along with the validation dataset
    - plotting: plotting is done on trained model both the training and validation curve.
    - saving: model is saved using "tf.keras.models.save_model" which makes it easy to run the "tensorflow_serving" server
    - prediction: convert the domain to numerical sequence with same length --> load model --> predict 


## Document any other approaches considered.
 - LSTM approach(https://www.elastic.co/blog/using-deep-learning-detect-dgas)
 - BERT (Since BERT is very good in NLP problems)






# Problem statement

Take a domain name as user input and find out if it is a valid domain or an auto-generated
domain using machine learning approach.
1. The domain name can be 255 characters long
2. The domain name will contain one TLD (.com, .ca, .io) at the end

The input to the model can be Alexa top 1 million sites (links given below).
The solution needs to be able to detect domains from at least 2 malware families from the link
given below.

# E.g.
● google.com is valid
● bind.com is valid
● lkhylm0mhyfuhg.ddns.net is DGA (ignore ddns.net)
● aktklyvbiu.com is DGA
● btpnxlsfdqbhzazyx.net is DGA

# Solution delivery
● Document the approach, pros and cons with the selected approach.
● Document any other approaches considered.
● A complete code implementation is not required though the differences between benign
and malicious samples should be made clear with the selected approach.
