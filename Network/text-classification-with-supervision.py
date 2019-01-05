import os
from pathlib import Path
from random import shuffle

import numpy as np
import pandas as pd
import sklearn.datasets as skds
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, BatchNormalization, Dropout, K
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

# Source file directory

fileDir = os.path.dirname(os.path.abspath(__file__))
# parentDir = os.path.dirname(fileDir)
parentDir = "C:\\Users\\mateu\\Desktop\\PoprawneCalosc\\PoprawneCalosc"
path_train = parentDir + "/Data"
path_to_labels = parentDir + "/Tag/Labelki"
cateogories = set()
d = {}
# Params
num_labels = 97
vocab_size = 3000
batch_size = 150
max_len = 300
mode = 'test'  # ['train', 'test']
read_from_csv = False



def find_categories(filename):


    text = Path(path_to_labels + "/" + filename).read_text()
    res = [int(i) for i in text.split()]
    return find_categories_with_array(res)


def find_categories_with_array(array):
    current_text_categories = []
    file = open("finalCategories.txt", "r")
    all_categories = [i for i in file.read().split()]
    for idx, value in enumerate(array):
        if value == 1:
            current_text_categories.append(all_categories[idx])

    return current_text_categories


def get_weights():
    keys = list(d.values())
    max_val = sum(keys)
    class_weights = {}
    for idx, value in enumerate(keys):
        class_weights[idx] = (int(max_val) / (int(value) * 10))
    return class_weights


# TODO Fmiara  Done
# TODO maxlen test  Done
# TODO sam abstract
# TODO Gotowe embedingi
def get_features(text_series):
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=max_len)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if not read_from_csv:
    files_train = skds.load_files(path_train, load_content=False)
    label_index = files_train.target
    label_names = files_train.target_names
    labelled_files = files_train.filenames

    data_tags = ["filename", "Text", 'Tags']
    data_list = []

    # Read and add data from file to a list
    print("Preparing dataset...")
    i = 0
    for f in labelled_files:
        data_list.append((f, Path(f).read_text(), find_categories(os.path.basename(f))))
        i += 1
    shuffle(data_list)
    # We have training data available as dictionary filename, category, data
    data = pd.DataFrame.from_records(data_list, columns=data_tags)

    print("Saving data frame to csv...")
    header = ["filename", "Text", "Tags"]
    data.to_csv("data.csv", index=False, header=header)
else:
    data = pd.read_csv("data.csv")
# datahehe = pd.read_csv("data.csv")

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
print(train_tags)
labels = multilabel_binarizer.classes_
y_train = multilabel_binarizer.transform(train_tags)
y_test = multilabel_binarizer.transform(test_tags)
if mode == 'train':
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9000)

    filter_length = 300

    model = Sequential()
    model.add(Embedding(2000, 20, input_length=max_len))
    # model.add(Dropout(0.1))
    model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
    model.add(BatchNormalization())
    model.add(GlobalMaxPool1D())
    model.add(Dense(num_labels, activation='sigmoid'))

    # Old model
    # model = Sequential()
    # model.add(Embedding(2000, 20, input_length=150))
    # model.add(Dropout(0.15))
    # model.add(GlobalMaxPool1D())
    # model.add(Dense(num_labels, activation='sigmoid'))

    model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=[f1, 'categorical_accuracy', 'accuracy'])
    tensor_board = TensorBoard(log_dir=fileDir + "/logs/conv/nope")
    checkpoint = ModelCheckpoint('checkpointweight.hdf5', monitor='val_loss', save_best_only=True)
    callbacks = [
        tensor_board,
        checkpoint
    ]

    # TODO writing and reading model

    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=callbacks
                        )
    model.model.save('my_model.h5')
    model.model.save_weights('wei.h5')
    # Deletes the existing model
else:
    # load json and create model
    model = load_model('my_model.h5', custom_objects={'f1': f1})
    model.model.load_weights('wei.h5')
    print("Loaded model from disk")

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size,
                       verbose=1
                       )
print(score)

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0
    predicted_label = find_categories_with_array(prediction[0])
    print(test_files_names.iloc[i])
    print(test_tags.iloc[i])
    print(predicted_label)
