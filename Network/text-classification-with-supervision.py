import os
from pathlib import Path
from random import shuffle

import numpy as np
import pandas as pd
import sklearn.datasets as skds
from keras.callbacks import TensorBoard
from keras.engine.saving import load_model, model_from_json
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPool1D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

# Source file directory
from sklearn.utils import class_weight

fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)
path_train = parentDir + "/Dane1"
path_labels = parentDir + "/Data"
path_to_labels = parentDir + "/Data/Labelki"

d = {}
with open("finalCategories.txt", "r") as file:
    for line in file:
        key, value = line.strip().split(" ")
        d[key] = int(value)


def find_categories(filename):
    text = Path(path_to_labels + "/" + filename).read_text()
    res = [int(i) for i in text.split()]
    return find_categories_with_array(res)


def find_categories_with_array(array):
    current_text_categories = []
    for idx, value in enumerate(array):
        if value == 1:
            current_text_categories.append(list(d.keys())[idx])

    return current_text_categories


def get_weights():
    keys = list(d.values())
    max_val = sum(keys)
    class_weights = {}
    for idx, value in enumerate(keys):
        class_weights[idx] = (int(max_val) / int(value))
    return class_weights

def get_features(text_series):
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=150)

# Params
num_labels = 97
vocab_size = 1500
batch_size = 100
mode = 'train'  # ['train', 'test']

files_train = skds.load_files(path_train, load_content=False)
label_index = files_train.target
label_names = files_train.target_names
labelled_files = files_train.filenames

data_tags = ["filename", "Text", 'Tags']
data_list = []

# Read and add data from file to a list
print("Creating dataset...")
i = 0
for f in labelled_files:
    data_list.append((f, Path(f).read_text(), find_categories(os.path.basename(f))))
    i += 1
shuffle(data_list)
# We have training data available as dictionary filename, category, data
data = pd.DataFrame.from_records(data_list, columns=data_tags)

print("Tokenizing...")
train_size = int(len(data) * .8)

train_texts = data['Text'][:train_size]
train_tags = data['Tags'][:train_size]
train_files_names = data['filename'][:train_size]

test_texts = data['Text'][train_size:]
test_tags = data['Tags'][train_size:]
test_files_names = data['filename'][train_size:]

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_texts)

x_train = get_features(train_texts)
x_test = get_features(test_texts)

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(train_tags)
labels = multilabel_binarizer.classes_
y_train = multilabel_binarizer.transform(train_tags)
y_test = multilabel_binarizer.transform(test_texts)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9000)

filter_length = 300

model = Sequential()
model.add(Embedding(3000, 20, input_length=150))
model.add(Dropout(0.1))
model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
model.add(BatchNormalization())
model.add(GlobalMaxPool1D())
model.add(Dense(num_labels, activation='sigmoid'))

# Old model
# model = Sequential()
# model.add(Dense(5000, activation='relu', input_dim=150))
# model.add(Dropout(0.1))
# model.add(Dense(600, activation='relu'))
# model.add(Dropout(0.1))
# model.add(GlobalMaxPool1D())
# model.add(Dense(num_labels, activation='sigmoid'))

model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['categorical_accuracy', 'accuracy'])

callbacks = [
    TensorBoard(log_dir=fileDir + "/logs/weight")
]

# TODO writing and reading model
if mode == 'train':
    history = model.fit(x_train, y_train,
                        epochs=35,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=callbacks
                        )
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
else:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size,
                           verbose=1
                           )
    print(score)

    for i in range(10):
        prediction = loaded_model.predict(np.array([x_test[i]]))
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        predicted_label = find_categories_with_array(prediction[0])
        print(test_files_names.iloc[i])
        print(test_tags.iloc[i])
        print(predicted_label)
