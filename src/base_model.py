import os
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from utils import (
    get_main_data,
    get_main_audio,
    filter_df,
    split_people,
    to_categorical,
    DEBUG,
    get_wav,
    to_mfcc,
    make_segments,
    extract_acoustic_features, 
    extract_plp,
    extract_prosodic_features,
)

# Load metadata
df = pd.read_csv('native_bio_metadata.csv')
filtered_df = filter_df(df)
# Train test split
X_train, X_test, y_train, y_test = split_people(filtered_df)
y_test = y_test.apply(lambda x: x.split('\n')[0])
y_train = y_train.apply(lambda x: x.split('\n')[0])

# drop the sicilian accent
# X_train = X_train[y_train != 'sicilian']
# y_train = y_train[y_train != 'sicilian']
# X_test = X_test[y_test != 'sicilian']
# y_test = y_test[y_test != 'sicilian']

# Get statistics
train_count = Counter(y_train)
test_count = Counter(y_test)


print(f"Train Count: {train_count}")
print(f"Test Count: {test_count}")
print("Entering main...")

# acc_to_beat = test_count.most_common(1)[0][1] / float(np.sum(list(test_count.values())))
# To categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# if DEBUG:
#     print('Extracting Features....')

# with ThreadPoolExecutor() as pool:
#     X_prosodic = pool.map(extract_prosodic_features, X_train)
#     X_acoustic = pool.map(extract_acoustic_features, X_train)
#     X_plp = pool.map(extract_plp, X_train)

# Get resampled wav files using multiprocessing
if DEBUG:
    print('Loading wav files....')
with ThreadPoolExecutor() as pool:
    X_train = pool.map(get_wav, X_train)
    X_test = pool.map(get_wav, X_test)

# Convert to MFCC
if DEBUG:
    print('Converting to MFCC....')
with ThreadPoolExecutor() as pool:
    X_train = pool.map(to_mfcc, X_train) 
    X_test = pool.map(to_mfcc, X_test)
    
# create segments
X_train, y_train = make_segments(X_train, y_train)
X_test, y_test = make_segments(X_test, y_test)

# create numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# input shape
rows = X_train[0].shape[0]
cols = X_train[0].shape[1]
num_classes = len(y_train[0])

input_shape = (rows, cols, 1)
print(rows, cols, num_classes)

# create a cnn model to train on the mfcc data
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(num_classes, activation='softmax'),
    ]
)

# compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# print the model summary
# print(model.summary())

# train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# save the model
model.save('files/base_model.keras')

# evaluate the model and plot confusion matrix
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot confusion matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# add labels
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
# save the confusion matrix
plt.savefig('files/base_model_confusion_matrix.png')
plt.colorbar()
plt.show()
