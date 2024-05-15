#from comet_ml import ConfusionMatrix, Experiment

import multiprocessing
import time
from collections import Counter, OrderedDict
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import os
from comet_ml import ConfusionMatrix, Experiment
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from multiprocessing import Pool, cpu_count
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.sequence import pad_sequences

#from constant_dir.constants import AUDIOS_INFO_FILE_NAME

#experiment = Experiment()

"""Parameters to adjust"""
LANG_SET = 'US_UK_CA_64mel_'  # what languages to use / fr_it_sp
OUTPUT_DIR: str = "output/"
CSV_FILE_PATH: str = "bio_metadata.csv"
NATIVE_FILE_PATH: str = "native_bio_metadata.csv"
NON_NATIVE_FILE_PATH: str = "non_native_bio_metadata.csv"
NATIVE_LANGUAGES: list[str] = ['uk', 'usa', 'canada']
NON_NATIVE_LANGUAGES: list[str] = [
    'australia',
    'new zealand',
    'ireland',    
    'singapore',  
    'south',     
    'africa',   
    'jamaica',    
    'scotland',   
    'islands',
]
DATASET_DIR: str = "data/"
NATIVE_DIR: str = "data/native/"
NATIVE_COMBINED_DIR: str = "data/native_combined/"
NON_NATIVE_DIR: str = "data/non_native/"
AUDIO_DATA_DIR: str = "data/audio/"
AUDIO_FILE_PATH: str = "data/audio/{}.wav"
FEATURES = 'fbe'  # mfcc / f0 / cen / rol / chroma / rms / zcr / fbe [Feature types] mfcc_f0_cen_rol_chroma_rms_zcr
MAX_PER_LANG = 80  # maximum number of audios of a language

UNSILENCE = False

WIN_LENGTH_MS = 25  # ms / 25
OVERLAP_MS = 10  # ms / 10

SAMPLE_RATE = 22050  # 22050 / 16000 [Hz]
HOP_LENGTH = int(SAMPLE_RATE * 0.001 * OVERLAP_MS)  # [10 ms overlap]
WIN_LENGTH = int(SAMPLE_RATE * 0.001 * WIN_LENGTH_MS)  # [25 ms window length]
# N_FFT = int(SAMPLE_RATE * 0.001 * WIN_LENGTH)  # [25 ms window length]
FRAME_SIZE = 75  # 30 / 50 / 70 / 100 / 150 / 200 / 300 / 500 [Size of feature segment]

MEL_S_LOG = False

selection_method = 'UNIVARIATE'  # PCE / UNIVARIATE
SCORE_FUNC = f_classif  # f_classif / mutual_info_classif [score function for univariate  feature selector]
NUM_OF_FEATURES = 3 # [number of optimal features to work with]
SELECT_FEATURES = False  # [whether or not use feature selection method]
CHECK_DATASETS = False

EPOCHS = 60  # [Number of training epochs]
BATCH_SIZE = 64  # size of mini-batch used
KERNEL_SIZE = (3, 3)  # (3, 3) (5, 5)
POOL_SIZE = (3, 3)  # (2, 2) (3, 3)
DROPOUT = 0.1  # 0.5 for mfcc CNN
BASELINE = 1.0
MIN_DELTA = .01  # .01
PATIENCE = 10  # 10
N_MELS = 64  # [number of filters for a mel-spectrogram]


def filter_df(df):
    """
    Filters the DataFrame to include only entries for native languages specified.
    This version removes the cap on the number per language.
    :param df: DataFrame - Unfiltered audio files DataFrame
    :return: DataFrame - Filtered DataFrame containing only native language entries
    """
    filtered_data_df = pd.DataFrame()
    for lang in NATIVE_LANGUAGES:
        filtered_data = df[df['english_residence'].isin([lang])]
        filtered_data_df = pd.concat([filtered_data_df, filtered_data])

    return filtered_data_df




def load_and_resample_audio(filename, target_sr=22050):
    try:
        file_path = f'./data/audio/{filename}.wav'
        y, sr = librosa.load(file_path, sr=None)  # Load with original sample rate
        y_resampled = librosa.core.resample(y=y, orig_sr=sr, target_sr=target_sr, scale=True)
        return y_resampled, target_sr
    except Exception as e:
        logger.error(f"Error loading wav: {filename} - {e}")
        return None, None
def extract_features(filename):
    """
    Extracts features from audio files.
    Different kinds of features are concatenated subsequently.
    :param filename: The basename of the audio file, expected to find it under './data/audio/'
    :return: (numpy.ndarray) feature matrices (columns == FRAME_SIZE, rows == number of features),
             or None if an error occurs during loading.
    """
    # Use the new function to load and resample the audio
    y, sr = load_and_resample_audio(filename, target_sr=SAMPLE_RATE)
    if y is None:  # If loading failed, return None
        return None

    # Continue with feature extraction only if audio was successfully loaded
    s, _ = librosa.magphase(librosa.stft(y, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
    features = []
    if 'mfcc' in FEATURES:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        features.append(mfccs)
    if 'f0' in FEATURES:
        f0 = librosa.yin(y, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'), sr=sr, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        features.append(f0)
    if 'cen' in FEATURES:
        spectral_centroid = librosa.feature.spectral_centroid(y, sr=sr, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        features.append(spectral_centroid)
    if 'rol' in FEATURES:
        spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        features.append(spectral_rolloff)
    if 'chroma' in FEATURES:
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        features.append(chromagram)
    if 'rms' in FEATURES:
        rms = librosa.feature.rms(y=y)[0]
        features.append(rms)
    if 'zcr' in FEATURES:
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=WIN_LENGTH * 2, hop_length=HOP_LENGTH)
        features.append(zcr)
    if 'fbe' in FEATURES:
        mel_s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, power=1.0)
        if MEL_S_LOG:
            mel_s = librosa.power_to_db(mel_s)
        features.append(mel_s)

    logger.debug('Concatenating extracted features...')
    if features:
        features = np.vstack(features)
        logger.debug(f'Shape of concatenated features: {features.shape}')
        return features
    else:
        return None

# Usage in your main code or preprocessing step
def preprocess_audio_files(filenames):
    """
    Preprocesses a list of filenames by extracting features.
    Handles errors in audio file loading.
    :param filenames: List of audio file names
    :return: List of feature vectors
    """
    features = []
    for filename in filenames:
        feature = extract_features(filename)
        if feature is not None:
            features.append(feature)
        else:
            logger.error(f"Skipping file {filename} due to load error")
    return features


def normalize_feature_vectors(feature_vectors):
    """
    Normalizes features presented by a vector (e.g. Mel-Cepstral coefficients, Mel-spectrogram).
    One vector corresponds to an audio segment of WIN_LENGTH length.
    :param feature_vectors: (numpy.ndarray) Vectors of features extracted from an audio file.
    :return: (numpy.ndarray) List of normalized vectors of features
    """
    mean = np.mean(feature_vectors.T, axis=0, dtype=np.float64)
    std = np.std(feature_vectors, dtype=np.float64)
    feature_vectors_normalized = []
    for i in range(feature_vectors.shape[1]):
        feature_vectors_normalized.append(np.subtract(feature_vectors[:, i], mean) / std)
    feature_vectors_normalized = np.array(feature_vectors_normalized)
    return feature_vectors_normalized.T


def normalize_scalar_feature(feature_vector):
    """
    Normalizes scalar features (e.g. spectral roll-off, F0, etc.
    Each feature is extracted from an audio segment of WIN_LENGTH length.
    :param feature_vector: (numpy.ndarray) Vector of scalar features
    :return: (numpy.ndarray) List of normalized features
    """
    mean = np.mean(feature_vector, dtype=np.float64)
    std = np.std(feature_vector, dtype=np.float64)
    feature_vector_normalized = (feature_vector - mean) / std
    return feature_vector_normalized


def derive_mfcc(audio_file, y):
    """
    Derives Mel-Cepstral coefficients from each frame of an audio file.
    Coefficients are normalized for each audio file to deal with
    the difference in volume and background noise.
    :param audio_file: (String) Relative audio file name
    :param y: (numpy.ndarray) Loaded and resampled at SAMPLE_RATE audio file
    :return: (numpy.ndarray) Vectors of normalized MFCC
    """
    logger.debug(f'Extracting MFCC for {audio_file}...')
    '''if 'energy' in LANG_SET:
        mfcc = python_speech_features.mfcc(signal=y, samplerate=SAMPLE_RATE, winlen=WIN_LENGTH / SAMPLE_RATE,
                                           winstep=HOP_LENGTH / SAMPLE_RATE, appendEnergy=True, numcep=14,
                                           winfunc=hann, preemph=0.0, ceplifter=0, nfilt=128, lowfreq=0,
                                           highfreq=None, nfft=2048).T
    if 'log' in LANG_SET:
        mel_s = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
                                               win_length=WIN_LENGTH, power=2.0)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_s), n_mfcc=13, hop_length=HOP_LENGTH,
                                    win_length=WIN_LENGTH)
                                    
    else: '''

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mfcc_normalized = normalize_feature_vectors(mfcc)
    return mfcc_normalized


def derive_mel_s(audio_file, y):
    """
    Derives Mel-Spectrogram of amplitude from each frame of an audio file.
    Coefficients are normalized for each audio file to deal with
    the difference in volume and background noise.
    :param audio_file: (String) Relative audio file name
    :param y: (numpy.ndarray) Loaded and resampled at SAMPLE_RATE audio file
    :return: (numpy.ndarray) Vectors of normalized mel-spectrograms
    """
    logger.debug(f'Extracting Mel-spectrogram for {audio_file}...')
    mel_s = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
                                           win_length=WIN_LENGTH, power=1.0)
    if MEL_S_LOG:
        mel_s = librosa.power_to_db(mel_s)
    mel_s_normalized = normalize_feature_vectors(mel_s)
    return mel_s_normalized


def derive_f0(audio_file, y):
    """
    Derives fundamental frequencies from each frame of an audio file.
    Coefficients are normalized for each audio file to deal with
    the difference in volume and background noise.
    :param audio_file: (String) Relative audio file name
    :param y: (numpy.ndarray) Loaded and resampled at SAMPLE_RATE audio file
    :return: (numpy.ndarray) Vector of normalized fundamental frequencies
    """
    logger.debug(f'Extracting fundamental frequency for {audio_file}...')
    f0 = librosa.yin(y, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'), sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                     win_length=WIN_LENGTH)
    f0_normalized = normalize_scalar_feature(f0)
    return f0_normalized


def derive_spectral_centroid(audio_file, y):
    """
    Derives spectral centroid from each frame of an audio file.
    Coefficients are normalized for each audio file to deal with
    the difference in volume and background noise.
    :param audio_file: (String) Relative audio file name
    :param y: (numpy.ndarray) Loaded and resampled at SAMPLE_RATE audio file
    :return: (numpy.ndarray) Vector of normalized spectral centroids
    """
    logger.debug(f'Extracting spectral centroid for {audio_file}...')
    spectral_centroid = librosa.feature.spectral_centroid(y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                          win_length=WIN_LENGTH)
    spectral_centroid_normalized = normalize_scalar_feature(spectral_centroid)
    return spectral_centroid_normalized


def derive_spectral_rolloff(audio_file, y):
    """
    Derives spectral centroid from each frame of an audio file.
    Coefficients are normalized for each audio file to deal with
    the difference in volume and background noise.
    :param audio_file: (String) Relative audio file name
    :param y: (numpy.ndarray) Loaded and resampled at SAMPLE_RATE audio file
    :return: (numpy.ndarray) Vector of normalized spectral roll-off values
    """
    logger.debug(f'Extracting spectral rolloff for {audio_file}...')
    rolloff = librosa.feature.spectral_rolloff(y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    rolloff_normalized = normalize_scalar_feature(rolloff)
    return rolloff_normalized


def derive_chromagram(audio_file, y):
    """
    Derives N chroma bins from each frame of an audio file.
    Coefficients are normalized for each audio file to deal with
    the difference in volume and background noise.
    :param audio_file: (String) Relative audio file name
    :param y: (numpy.ndarray) Loaded and resampled at SAMPLE_RATE audio file
    :return: (numpy.ndarray) Vectors of normalized chroma bins of an audio file
    """
    logger.debug(f'Extracting chromagram for {audio_file}...')
    chromagram = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    chromagram_normalized = normalize_feature_vectors(chromagram)
    return chromagram_normalized


def derive_rms(audio_file, s):
    """
    Derives root-mean-square (RMS) value from each frame of an audio file.
    Coefficients are normalized for each audio file to deal with
    the difference in volume and background noise.
    :param audio_file: (String) Relative audio file name
    :param s: (numpy.ndarray) magnitudes (S) of a Spectrogram
    :return: (numpy.ndarray) Vector of normalized RMS values
    """
    logger.debug(f'Extracting chromagram for {audio_file}...')
    rms = librosa.feature.rms(S=s)[0]
    rms_normalized = normalize_scalar_feature(rms)
    return rms_normalized


def derive_zcr(audio_file, y):
    """
    Derives zero-crossing rate from each frame of an audio file.
    Coefficients are normalized for each audio file to deal with
    the difference in volume and background noise.
    :param audio_file: (String) Relative audio file name
    :param y: (numpy.ndarray) Loaded and resampled at SAMPLE_RATE audio file
    :return: (numpy.ndarray) Vector of normalized ZCR
    """
    logger.debug(f'Extracting ZCR for {audio_file}...')
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH, frame_length=WIN_LENGTH * 2)
    zcr_normalized = normalize_scalar_feature(zcr)
    return zcr_normalized


def split_into_matrices(feature_vectors, labels):
    """
    Makes segments of vectors of features
    and attaches them to the corresponding labels.
    :param feature_vectors: vectors of features
    :param labels: list of labels
    :return: (tuple) Matrices with corresponding labels
    """
    segments = []
    seg_labels = []
    for feature_vector, label in zip(feature_vectors, labels):
        for frame_start in range(0, int(feature_vector.shape[1] / FRAME_SIZE)):
            segments.append(feature_vector[:, frame_start * FRAME_SIZE:(frame_start + 1) * FRAME_SIZE])
            seg_labels.append(label)
    return segments, seg_labels


def create_segments_after_selection(data_arrays):
    """
    Splits selected features into matrices
    :param data_arrays:
    :return: matrices of features
    """
    segments_arrays = ()
    for data_array in data_arrays:
        segments = []
        logger.debug(f'\nShape of data before segmenting: {data_array.shape}')
        for element in data_array:
            segments.append(element.reshape(NUM_OF_FEATURES, FRAME_SIZE))
        logger.debug(f'Shape of segmented data: {np.array(segments).shape}\n')
        segments_arrays = segments_arrays + (np.array(segments),)
    return segments_arrays






def preprocess_new_data(audio_filenames, corresponding_languages):
    logger.info(f'Processing {len(audio_filenames)} audio files.')
    
    if not audio_filenames:
        logger.error("No audio filenames provided.")
        return None  # Consider what default values to return instead of None

    # Transform y into a categorical format
    le = LabelEncoder()
    y_categorical = to_categorical(le.fit_transform(corresponding_languages))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    features = pool.map(extract_features, audio_filenames)
    pool.close()
    pool.join()

    # Filter out None values
    filtered_features = [f for f, y in zip(features, y_categorical) if f is not None]
    filtered_y_categorical = [y for f, y in zip(features, y_categorical) if f is not None]

    if not filtered_features:
        logger.error("No valid audio files were processed. All features are None.")
        return None  # Consider what default values to return instead of None

    # Assume the rest of the processing occurs here and is successful
    # Dummy return values as placeholders for actual processed data
    x_train, x_test, y_train, y_test = train_test_split(filtered_features, filtered_y_categorical, test_size=0.25)
    train_count = Counter([np.argmax(y) for y in y_train])
    test_count = Counter([np.argmax(y) for y in y_test])
    languages_mapping = get_classes_map(y_categorical, corresponding_languages)

    return x_train, x_test, y_train, y_test, train_count, test_count, languages_mapping



def get_classes_map(y, y_raw):
    """
    :param y: binary representation of labels
    :param y_raw: list of languages in String form
    :return (OrderedDict): language to binary correspondence
    """
    classes = {}
    while len(classes) < len(Counter(y_raw)):
        for raw, category in zip(y_raw, y):
            classes[np.argmax(category)] = raw
    ordered_classes = OrderedDict(sorted(classes.items()))
    return ordered_classes


def save_input_data_to_files(x_train, x_test, y_train, y_test, train_count, test_count, classes):
    """
    Creates 2 files:
    - file with training and testing sets saved
    - file with information about classes distribution
    :param x_train: training feature matrices
    :param x_test: testing feature matrices
    :param y_train: corresponding training labels
    :param y_test: corresponding testing labels
    :param train_count: distribution by classes in training set
    :param test_count: distribution by classes in testing set
    :param classes: language to binary correspondence
    :return:
    """
    with open(info_data_npy, 'wb') as f:
        np.save(f, np.array([train_count, test_count]))
        np.save(f, classes)
    with open(features_npy, 'wb') as f:
        np.save(f, x_train)
        np.save(f, y_train)
        np.save(f, x_test)
        np.save(f, y_test)


def open_preprocessed_data():
    """
    Retrieves training and testing sets
    and information about classes distribution
    saved before from files.
    :return: (tuple) training samples, testing samples,
    training labels, testing labels,
    distribution by classes in training set,
    distribution by classes in testing set,
    language to binary correspondence
    """
    with open(features_npy, 'rb') as f:
        x_train = np.load(f)
        y_train = np.load(f)
        x_test = np.load(f)
        y_test = np.load(f)
    with open(info_data_npy, 'rb') as f:
        counts = np.load(f, allow_pickle=True)
        train_count = counts[0]
        test_count = counts[1]
        classes = np.load(f, allow_pickle=True).item()
    return x_train, x_test, y_train, y_test, train_count, test_count, classes


class TerminateOnBaseline(Callback):
    """
    Callback that terminates training when
    either accuracy or val_acc reaches
    a specified baseline
    """

    def __init__(self, monitor='accuracy', baseline=BASELINE):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get(self.monitor)
        if accuracy is not None:
            if accuracy >= self.baseline:
                logger.debug(f'Epoch {epoch}: Reached baseline, terminating training...')
                self.model.stop_training = True


class TimeHistory(Callback):
    """
    Callback that saves duration of every training epoch into list.
    """

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def compare_sets(x_1, x_2):
    """
    :param x_1: (numpy.ndarray) list of 2-D numpy arrays; training set
    :param x_2: (numpy.ndarray) list of 2-D numpy arrays; testing set
    :return: String (how many occurances have been found)
    """
    equal_matrices_num = 0
    indices_to_remove = []
    for matrix_idx, x_2_matrix in enumerate(x_2):
        for x_1_matrix in x_1:
            if (x_1_matrix == x_2_matrix).all():
                equal_matrices_num += 1
                indices_to_remove.append(matrix_idx)
                break
    return f'Number of equal matrices in sets: {equal_matrices_num}.'


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    plt.show()


def train_model(x_train, y_train, x_validation, y_validation):
    """
    Prepares data for training a 2D CNN model,
    builds a model,
    performs a model training,
    plots accuracy and loss changes during training.
    :param x_train: (numpy.ndarray) list of feature matrices for training the network
    :param y_train: (numpy.ndarray) list of binary labels training the network
    :param x_validation: (numpy.ndarray) list of feature matrices for testing the network
    :param y_validation: (numpy.ndarray) list of binary labels testing the network
    :return: Trained model
    """
    if CHECK_DATASETS:
        logger.debug('Checking whether train and test sets are different...')
        logger.debug(f'X train compared with itself. {compare_sets(x_train, x_train)}')
        logger.debug(f'X validation compared with itself. {compare_sets(x_validation, x_validation)}')
        logger.debug(f'X train compared with x validation. {compare_sets(x_train, x_validation)}')

    logger.debug('Getting data dimensions...')

    rows = x_train[0].shape[0]
    cols = x_train[0].shape[1]
    assert x_train[0].shape == x_validation[0].shape
    logger.debug('Train and validation matrices are of same dimension...')

    train_samples_num = x_train.shape[0]
    val_samples_num = x_validation.shape[0]
    assert train_samples_num == y_train.shape[0] and val_samples_num == y_validation.shape[0]
    logger.debug('X and Y have the same number of samples...')

    num_classes = y_train[0].shape[0]

    logger.debug(f'Input matrix rows: {rows}')
    logger.debug(f'Input matrix columns: {cols}')
    logger.debug(f'Num. of classes: {num_classes}')

    logger.debug('Reshaping input data...')

    input_shape = (rows, cols, 1)
    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
    x_validation = x_validation.reshape(x_validation.shape[0], rows, cols, 1)
    logger.debug(f'Input data shape: {input_shape}')

    model = build_model(input_shape, num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    logger.debug(f'Creating a condition for stopping training if accuracy does not change '
                 f'at least {MIN_DELTA * 100}% over {PATIENCE} epochs')

    es = EarlyStopping(monitor='accuracy', min_delta=MIN_DELTA, patience=PATIENCE, verbose=1, mode='auto',
                       restore_best_weights=True)
    # es_baseline = TerminateOnBaseline(monitor='accuracy', baseline=BASELINE)
    time_history = TimeHistory()

    logger.debug('Adding image generator for data augmentation...')
    data_generator = ImageDataGenerator(width_shift_range=0.2)

    logger.debug('Training model...')
    history = model.fit(data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=x_train.shape[0] / BATCH_SIZE
                        , epochs=EPOCHS,
                        callbacks=[es, time_history], validation_data=(x_validation, y_validation))
    epoch_av_time = round(np.mean(time_history.times), 2)

    logger.debug('Model trained.')
    logger.info(f'Average epoch time: {epoch_av_time}')
    logger.debug('Plotting accuracy and loss...')

    plot_history(history)

    return model


def build_model(input_shape, num_classes):
    """
    Builds a 2D CNN model.
    :param input_shape: (tuple) shape of input data
    to pass to the 1st convolutional layer
    :param num_classes: (Int) number of classes for classification
    :return: Built Keras 2D CNN model
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=KERNEL_SIZE, activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Conv2D(64, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(DROPOUT))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# from comet_ml import ConfusionMatrix, Experiment

# Other imports remain unchanged

# Remove or comment out experiment initialization
# experiment = Experiment()

# Your existing Python code for processing, training, etc., remains here

def plot_history(history):
    """
    Plots how training and testing
    accuracy and loss change
    over the training process
    :param history: a model's training history
    :return:
    """
    fig, ax_loss = plt.subplots(constrained_layout=True)
    ax_acc = ax_loss.twinx()

    ax_loss.plot(history.history['loss'], label='train loss', color='#E43F04')
    ax_loss.plot(history.history['val_loss'], label='test loss', color='#FF9147')

    ax_acc.plot(history.history['accuracy'], label='train acc', color='#2201C7')
    ax_acc.plot(history.history['val_accuracy'], label='test acc', color='#0055FF')

    ax_loss.set_xlabel('epochs')
    ax_loss.set_ylabel('loss')
    ax_acc.set_ylabel('accuracy')

    ax_loss.legend(loc='upper left')
    ax_acc.legend(loc='lower left')

    plt.title('Model train vs validation')

    plt.show()
    # Comment out or adjust any Comet.ml specific logging
    # experiment.log_figure(figure=plt)

# Other functions remain unchanged



def one_hot_to_int(one_hot_arr):
    """
    Convert one-hot encoded data to Int
    :param one_hot_arr: list of one-hot encoded numbers
    :return: list of numbers represented as integers
    """
    return np.array([np.argmax(one_hot) for one_hot in one_hot_arr])


def select_features(x_train, y_train, x_test):
    """
    Performs features selection by flatenning
    feature matrices
    :param x_train: (numpy.ndarray) list of feature matrices used for training
    :param y_train: (numpy.ndarray) list of binary labels
    :param x_test: (numpy.ndarray) list of features matrices used for testing
    :return:
    """
    logger.debug('Performing feature selection...')
    logger.debug('[BEFORE SELECTION]')  # matrices won't pass for selection. Choose distinct vectors.
    logger.debug(f'X train shape: {x_train.shape}')
    logger.debug(f'y train shape: {y_train.shape}')
    logger.debug(f'X test shape: {x_test.shape}')

    y_train = one_hot_to_int(y_train)

    x_train = np.array([x_train.flatten() for x_train in x_train])
    x_test = np.array([x_test.flatten() for x_test in x_test])

    logger.debug('\n[AFTER SELECTION]')
    logger.debug(f'X train shape: {x_train.shape}')
    logger.debug(f'y train shape: {y_train.shape}')
    logger.debug(f'X test shape: {x_test.shape}')

    if selection_method == 'UNIVARIATE':
        selector = SelectKBest(score_func=SCORE_FUNC,
                               k=NUM_OF_FEATURES * FRAME_SIZE)  # k = number of features to choose
        selector.fit(x_train, y_train)
        logger.info(f'Feature selection score: [{selector.scores_}]')
    elif selection_method == 'PCE':
        selector = PCA(n_components=NUM_OF_FEATURES * FRAME_SIZE)
        selector.fit(x_train)
        logger.info(f'Explained Variance: {selector.explained_variance_ratio_}')
        logger.info(selector.components_)

    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test)

    return x_train_selected, x_test_selected


import os

def main():
    global LANG_SET, features_npy, info_data_npy

    if UNSILENCE:
        LANG_SET += '_unsilenced'
    training_languages_str = f'{LANG_SET}_{FRAME_SIZE}'

    Path(f'./features/{FEATURES}').mkdir(parents=True, exist_ok=True)
    Path(f'./testing_data/{FEATURES}').mkdir(parents=True, exist_ok=True)
    Path(f'./models/{FEATURES}').mkdir(parents=True, exist_ok=True)

    features_npy = f'./features/{FEATURES}/{training_languages_str}.npy'
    info_data_npy = f'./testing_data/{FEATURES}/{training_languages_str}.npy'
    model_file = f'./models/{FEATURES}/{training_languages_str}.h5'

    if not Path(features_npy).exists() or not Path(info_data_npy).exists():
        df = pd.read_csv(NATIVE_FILE_PATH)
        df = filter_df(df)
        # Ensure that audio file paths are correctly formed
        audio_filenames = [x for x in df['language_num']] # .apply(os.path.basename)

        corresponding_languages = df['language_num'].tolist()

        # Proceed if the lists are populated
        if audio_filenames and corresponding_languages:
            preprocess = preprocess_new_data(audio_filenames, corresponding_languages)
            x_train, x_test, y_train, y_test, train_count, test_count, languages_mapping = preprocess
        else:
            logger.error("Audio filenames or languages are missing or incorrectly formatted.")
            return
    else:
        x_train, x_test, y_train, y_test, train_count, test_count, languages_mapping = open_preprocessed_data()

    # Feature selection and model training as before...





    logger.debug('Selecting features...')

    if SELECT_FEATURES:
        x_train, x_test = select_features(x_train, y_train, x_test)
        x_train, x_test = create_segments_after_selection((x_train, x_test))

    logger.debug('Training model...')

    if not Path.exists(Path(model_file)):
        # Find the maximum sequence length separately for x_train and x_test
        max_length_train = max(len(x) for x in x_train)
        max_length_test = max(len(x) for x in x_test)

        # Find the overall maximum sequence length (consider handling variable lengths if needed)
        max_length = max(max_length_train, max_length_test)

        # Pad the sequences in x_train (check padding consistency and consider pre-allocation)
        x_train_padded = np.zeros((len(x_train), max_length,  # Pre-allocate if max_length is known
                                # Feature dimension based on your data (replace with actual dimension)
                                ), dtype=x_train[0].dtype)
        for i, x in enumerate(x_train):
            pad_width = max_length - len(x)
            x_train_padded[i, :len(x)] = x  # Copy the original sequence
            x_train_padded[i, len(x):] = np.pad(x[-1:], (0, pad_width), mode='constant')  # Pad with the last element

        # Pad the sequences in x_test (similar approach)
        x_test_padded = np.zeros((len(x_test), max_length,  # Pre-allocate if applicable
                                # Feature dimension based on your data
                                ), dtype=x_test[0].dtype)
        for i, x in enumerate(x_test):
            pad_width = max_length - len(x)
            x_test_padded[i, :len(x)] = x
            x_test_padded[i, len(x):] = np.pad(x[-1:], (0, pad_width), mode='constant')

        # Convert to numpy array
        trained_model = train_model(np.array(x_train_padded), np.array(y_train), np.array(x_test_padded), np.array(y_test))
        np.array(y_test)
        trained_model.summary()
        trained_model.save(model_file)
    else:
        trained_model = load_model(model_file)

    languages_classes_mapping = list(languages_mapping.values())

    logger.debug('Running model on testing set...')
    logger.debug(f'X train shape: {x_train.shape}')
    logger.debug(f'X test shape: {x_test.shape}')
    logger.debug(f'Y train shape: {y_train.shape}')
    logger.debug(f'Y test shape: {y_test.shape}')

    y_predicted = trained_model.predict_classes(x_test.reshape(x_test.shape + (1,)), verbose=1)
    y_test_bool = np.argmax(y_test, axis=1)

    logger.info(f'Metrics:\n{classification_report(y_test_bool, y_predicted, target_names=languages_classes_mapping)}')
    logger.debug('Printing statistics (training ans testing counters)...')
    logger.info(f'Training samples: {train_count}')
    logger.info(f'Testing samples: {test_count}')
    logger.debug('Displaying a confusion matrix, overall accuracy...')

    cm = ConfusionMatrix()
    cm.compute_matrix(y_test, y_predicted)
    cm.labels = languages_classes_mapping
    confusion_matrix = np.array(cm.to_json()['matrix'])

    #experiment.log_confusion_matrix(matrix=cm)

    logger.debug('Accuracy to beat = (samples of most common class) / (all samples)')
    acc_to_beat = np.amax(np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix))
    confusion_matrix_acc = np.sum(confusion_matrix.diagonal()) / float(np.sum(confusion_matrix))
    trained_model.evaluate(x_test.reshape(x_test.shape + (1,)), y_test)

    logger.info(f'Accuracy to beat: {acc_to_beat}')
    logger.info(f'Confusion matrix:\n {confusion_matrix}')
    logger.info(f'Accuracy: {confusion_matrix_acc}')
    logger.debug('Displaying the baseline, and whether it has been hit...')

    baseline_difference = confusion_matrix_acc - acc_to_beat
    if baseline_difference < 0:
        logger.info('Baseline has not been hit.')
    else:
        logger.info(f'Baseline score: {baseline_difference}')

    logger.debug('Showing languages to categorical mapping...')
    logger.info(f'Relation classes to categories: {languages_mapping}')

    y_predicted_prob = trained_model.predict(x_test.reshape(x_test.shape + (1,)), verbose=1)
    logger.info(y_predicted[:10])
    logger.info('PROB: ')
    logger.info(y_predicted_prob[:10])


if __name__ == '__main__':
    main()
