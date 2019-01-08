import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import sklearn.datasets as skds
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from custom_measures import f1
from utils import find_categories, find_categories_with_array

fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = "C:\\Users\\mateu\\Desktop\\Podzielone\\Podzielone"

path_test_data = parentDir + "/test/Data"
path_to_test_labels = parentDir + "/test/Tag/Labelki"

# Params
num_labels = 97
vocab_size = 3000
batch_size = 150
max_len = 300

def get_features(text_series):
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=max_len)


files_test = skds.load_files(path_test_data, load_content=False)
labelled_test_files = files_test.filenames

data_tags = ["filename", "Text", 'Tags']
data_list = []
data_test_list = []
print("Preparing dataset...")

for f in labelled_test_files:
    data_test_list.append((f, Path(f).read_text(), find_categories(os.path.basename(f), path_to_test_labels)))

test_data = pd.DataFrame.from_records(data_test_list, columns=data_tags)

print("Tokenizing...")

test_texts = test_data['Text']
test_tags = test_data['Tags']
test_files_names = test_data['filename']

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

x_test = get_features(test_texts)

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(test_tags)
labels = multilabel_binarizer.classes_
y_test = multilabel_binarizer.transform(test_tags)

model = load_model('caluski.hdf5', custom_objects={'f1': f1})
print("Loaded model from disk")

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size,
                       verbose=1
                       )
print(score)

for i in range(100):
    prediction = model.predict(np.array([x_test[i]]))
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0
    predicted_label = find_categories_with_array(prediction[0])
    print(test_files_names.iloc[i])
    print(test_tags.iloc[i])
    print(predicted_label)
