# !/usr/bin/env python

"""
    The main model can be run from here
    The training, testing and consuming the
    model can be done from here
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from model import MainModel
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS']= "0"


def get_all_audio():
    """
        Get the data from speech archive by running the function in the get_data.py in the utils folder
        Get the audio data by running the function in the get_audio.py in the utils folder
        This function will get the audio data from the speech archive and save it in the data folder
    """
    get_main_audio()


def get_all_data():
    """
        Get the data from speech archive by running the function in the get_data.py in the utils folder
        Get the audio data by running the function in the get_audio.py in the utils folder
        This function will get the audio data from the speech archive and save it in the data folder
    """
    _ = get_main_data()
    get_all_audio()


def model_workflow():
    # Load metadata
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
    
    if DEBUG:
        print('Extracting Features....')
    
    with ThreadPoolExecutor() as pool:
        X_prosodic = pool.map(extract_prosodic_features, X_train)
        X_acoustic = pool.map(extract_acoustic_features, X_train)
        X_plp = pool.map(extract_plp, X_train)

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
    
    # Create segments from MFCCs
    X_train, y_train = make_segments(X_train, y_train)
    X_validation, y_validation = make_segments(X_test, y_test)    
    # Create model
    model = MainModel(
        x_train=np.array(X_train),
        y_train=np.array(y_train),
        x_validation=np.array(X_validation),
        y_validation=np.array(y_validation)
    )
    
    # create model
    model.create_cnn_model()
    #model.create_transformer_model()
    # model summary
    model.model_summary()
    # plot model
    model.plot_model()
    #Train model
    model.train_cnn_model()
    #model.train_transformer_model()
    # save model
    model.save_model()
    # plot model summary
    model.plot_model_stats()
    # get accuracy with X_test and y_test
    accuracy_value = model.get_accuracy(X_validation, y_validation)
    # get confusion matrix with X_test and y_test
    eval_value = model.evaluate_model(X_validation, y_validation)
    print(f"Accuracy Value: {accuracy_value}")
    print(f"Eval Value: {eval_value}")
    # plot the confusion matrix
    model.plot_confusion_matrix()
    model.get_confusion_matrix(X_validation, y_validation)


def main():
    """
        The main function for the model
        train the model, test the model
        evaluate the model
        and consume the model from here
    """
    model_workflow()


if __name__ == "__main__":
    main()
