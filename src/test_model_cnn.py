import pandas as pd
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    filter_df,
    split_people,
    to_categorical,
    DEBUG,
    get_wav,
    to_mfcc,
    make_segments,
)

df = pd.read_csv('bio_metadata.csv')
filtered_df = filter_df(df)

# Train test split
X_train, X_test, y_train, y_test = split_people(filtered_df)
y_test = y_test.apply(lambda x: x.split('\n')[0])
y_train = y_train.apply(lambda x: x.split('\n')[0])

# drop the sicilian accent
X_train = X_train[y_train != 'sicilian']
y_train = y_train[y_train != 'sicilian']
X_test = X_test[y_test != 'sicilian']
y_test = y_test[y_test != 'sicilian']

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

# Get resampled wav files using multiprocessing
if DEBUG:
    print('Loading wav files....')
with ThreadPoolExecutor() as pool:
    X_test = pool.map(get_wav, X_test)

# Convert to MFCC
if DEBUG:
    print('Converting to MFCC....')
with ThreadPoolExecutor() as pool:
    X_test = pool.map(to_mfcc, X_test)
    
X_validation, y_validation = make_segments(X_test, y_test) 
X_validation = np.array(X_validation)
y_validation = np.array(y_validation)

# resize data
rows = X_validation[0].shape[0]
cols = X_validation[0].shape[1]
num_classes = len(y_validation[0])

# input image dimensions to feed into 2D ConvNet Input layer
input_shape = (rows, cols, 1)
X_validation = X_validation.reshape(X_validation.shape[0], rows, cols, 1)

# load model from output/model5.keras
model = load_model('output/model5.keras')

# print the model summary
# print(model.summary())

# test the model
score = model.evaluate(X_validation, y_validation, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot confusion matrix
y_pred = model.predict(X_validation)
y_pred = np.argmax(y_pred, axis=1)
y_validation = np.argmax(y_validation, axis=1)
cm = confusion_matrix(y_validation, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_count.keys(), yticklabels=train_count.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
# save image to output
plt.savefig('output/confusion_matrix_cnn.png')
plt.show() 

