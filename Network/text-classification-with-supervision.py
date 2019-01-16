import os
import pickle
from pathlib import Path

import pandas as pd
import sklearn.datasets as skds
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPool1D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

from custom_measures import f1
from utils import find_categories

fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = "C:\\Users\\mateu\\Desktop\\Podzielone\\Podzielone"
path_train = parentDir + "/train/Data"
path_to_labels = parentDir + "/train/Tag/Labelki"

# Params
num_labels = 97
vocab_size = 3000
batch_size = 200
max_len = 450


def get_features(text_series):
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=max_len)


files_train = skds.load_files(path_train, load_content=False)
labelled_files = files_train.filenames

data_tags = ["filename", "Text", 'Tags']
data_list = []
print("Preparing dataset...")
for f in labelled_files:
    data_list.append((f, Path(f).read_text(), find_categories(os.path.basename(f), path_to_labels)))

data = pd.DataFrame.from_records(data_list, columns=data_tags)

print("Tokenizing...")

train_texts = data['Text']
train_tags = data['Tags']
train_files_names = data['filename']

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_texts)

x_train = get_features(train_texts)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(train_tags)
labels = multilabel_binarizer.classes_
y_train = multilabel_binarizer.transform(train_tags)

filter_length = 300

model = Sequential()
model.add(Embedding(2000, 20, input_length=max_len))
model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
model.add(BatchNormalization())
model.add(GlobalMaxPool1D())
model.add(Dense(num_labels, activation='sigmoid'))
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=[f1, 'categorical_accuracy', 'accuracy'])


# Old model
# model = Sequential()
# model.add(Embedding(2000, 20, input_length=150))
# model.add(Dropout(0.15))
# model.add(GlobalMaxPool1D())
# model.add(Dense(num_labels, activation='sigmoid'))

tensor_board = TensorBoard(log_dir=fileDir + "/logs/conv/450-with-early-stopping")

checkpoint = ModelCheckpoint('checkpointweight.hdf5',
                             monitor='val_loss',
                             save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=5,
                               verbose=0,
                               mode='auto')
callbacks = [
    tensor_board,
    early_stopping,
    checkpoint
]

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks
                    )

score = model.evaluate(x_train, y_train,
                       batch_size=batch_size,
                       verbose=1
                       )
print(score)
model.save('caluski.hdf5')
