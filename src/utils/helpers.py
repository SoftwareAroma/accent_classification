# !/usr/bin/env python
"""
    This file contains the helper methods and functions 
    used by other parts of the code no 
    necessarily in this file
"""
import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from constants import RATE, N_MFCC, COL_SIZE
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS']= "0"

# if gpu is available, use it
if tf.config.list_physical_devices('GPU'):
    device = 'gpu'
else:
    device = 'cpu'


# def to_categorical(languages: list):
#     """
#         Converts list of languages into a binary class matrix
#         Args:
#             :param languages: list of languages
#         Return:
#             :return (numpy array): binary class matrix
#     """
#     lang_dict = {}
#     for index, language in enumerate(set(languages)):
#         lang_dict[language] = index
#     y = list(map(lambda x: lang_dict[x], languages))
#     return keras.utils.to_categorical(y, len(lang_dict))

def to_categorical(y):
    '''
        Converts list of languages into a binary class matrix
        :param y (list): list of languages
        :return (numpy array): binary class matrix
    '''
    lang_dict = {}
    for index,language in enumerate(set(y)):
        lang_dict[language] = index
    y = list(map(lambda x: lang_dict[x],y))
    return keras.utils.to_categorical(y, len(lang_dict))


def get_wav(filename):
    """
        Loads a wav file from disk and resamples to a target sample rate.

        Args:
            filename (str): Path to the wav file.

        Returns:
            numpy.ndarray: Down-sampled wav file (or None if an error occurs).
    """
    try:
        y, sr = librosa.load(f'data/audio/{filename}.wav')
        return librosa.core.resample(y=y, orig_sr=sr, target_sr=RATE, scale=True)
    except Exception as e:
        print(f"Error loading wav: {filename} - {e}")
        return None  # Or handle error differently


def to_mfcc(wav):  # Optional arguments for flexibility
    """
        Converts a wav file to Mel Frequency Ceptral Coefficients (MFCCs).
        Args:
            wav (numpy array): The wav form data.
            sr (int, optional): The sample rate of the audio. Defaults to None (use from data if available).
            n_mfcc (int, optional): The number of MFCC coefficients to extract. Defaults to None (use librosa's default).
        Returns:
            numpy.ndarray: A 2D numpy array containing the MFCC features.
        Raises:
            Exception: If an error occurs during processing.
    """
    try:
        return librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC)
    except Exception as e:
        print(f"Error converting wav to MFCC: {e}")
        return None  # Or handle error differently
    
def normalize_mfcc(mfcc):
    '''
        Normalize mfcc
        :param mfcc:
        :return:
    '''
    mms = MinMaxScaler()
    return(mms.fit_transform(np.abs(mfcc)))
    

def remove_silence(wav, thresh=0.04, chunk=5000):
    '''
        Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, the values will be removed
        :param wav (np array): Wav array to be filtered
        :return (np array): Wav array with silence removed
    '''

    tf_list = []
    for x in range(len(wav) / chunk):
        if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
            tf_list.extend([True] * chunk)
        else:
            tf_list.extend([False] * chunk)

    tf_list.extend((len(wav) - len(tf_list)) * [False])
    return(wav[tf_list])


def make_segments(mfccs,labels):
    '''
        Makes segments of mfccs and attaches them to the labels
        :param mfccs: list of mfccs
        :param labels: list of labels
        :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment_one(mfcc):
    '''
        Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
        :param mfcc (numpy array): MFCC array
        :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))


def create_segmented_mfccs(X_train):
    '''
        Creates segmented MFCCs from X_train
        :param X_train: list of MFCCs
        :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)


def extract_mfcc_lst(audio_segment, sr=22050, n_mfcc=13, mel_filter_bank=20) -> None:
    """
        Extracts MFCCs from a given audio segment.
        Args:
            :param audio_segment: A NumPy array of the audio data.
            :param sr: Sampling rate of the audio (default: 22050 Hz).
            :param n_mfcc: Number of MFCC coefficients to extract (default: 13).
            :param mel_filter_bank: Number of mel filters to use (default: 20).
        Returns:
            A NumPy array of MFCC features for the audio segment.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        audio_segment, sr=sr, n_mels=mel_filter_bank
    )
    mfcc_lst = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_spectrogram), 
        n_mfccs=n_mfcc
    )
    return mfcc_lst.T  # Transpose to have features as columns


def normalize_mfcc_lst(mfcc_lst: np.array, method: str = None) -> np.array:
    """
        Normalizes MFCC features.

        Args:
            :param mfcc_lst: A NumPy array of MFCC features.
            :param method: Normalization method (default: None) -> This will use the MinMaxScaler.
                    Supported methods: 'minmax', 'standard'

        Returns:
            :return: The normalized MFCC features.
    """
    if method == 'minmax':
        return (mfcc_lst - np.min(mfcc_lst)) / (np.max(mfcc_lst) - np.min(mfcc_lst))
    elif method == 'standard':
        return (mfcc_lst - np.mean(mfcc_lst)) / np.std(mfcc_lst)
    else:
        mms = MinMaxScaler()
        return mms.fit_transform(np.abs(mfcc_lst))


def reduce_dimensionality(mfcc_lst: np.array, n_components: int = 10) -> np.array:
    """
        Reduces the dimensionality of MFCC features using PCA.
        The number of principal components to retain is specified by n_components.

        Args:
            :param mfcc_lst: A NumPy array of MFCC features.
            :param n_components: Number of principal components to retain (default: 10).

        Returns:
            :return: The reduced MFCC features.
    """
    pca = PCA(n_components=n_components)
    reduced_mfcc_lst = pca.fit_transform(mfcc_lst)
    return reduced_mfcc_lst


def combine_segments(mfcc_segments: list, method='mean'):
    """
        Combines MFCC features from multiple segments.
        Args:
            :param mfcc_segments: A list of NumPy arrays of MFCC features, one for each segment.
            :param method: Method for combining features (default: 'mean').
                    Supported methods: 'mean', 'concatenate'

        Returns:
            :return: A NumPy array of combined MFCC features.
    """
    if method == 'mean':
        return np.mean(mfcc_segments, axis=0)
    elif method == 'concatenate':
        return np.concatenate(mfcc_segments, axis=1)
    else:
        raise ValueError(f"Invalid combination method: {method}")


def filter_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        Function to filter audio files based on df columns
        df column options: [age,age_of_english_onset,age_sex,birth_place,english_learning_method,
        english_residence,length_of_english_residence,native_language,other_languages,sex]
        :return (DataFrame): Filtered DataFrame
    """

    # if the dataframe['native_language'] has arabic, mandarin, or english then
    arabic = dataframe[dataframe.native_language == 'arabic']
    mandarin = dataframe[dataframe.native_language == 'mandarin']
    english = dataframe[dataframe.native_language == 'english']    
    mandarin = mandarin[mandarin.length_of_english_residence < 10]
    arabic = arabic[arabic.length_of_english_residence < 10]
    # use concat to add the dataframes together
    return pd.concat(
        [
            dataframe,
            mandarin,
            arabic,
            english,
        ],
        ignore_index=True
    )


def split_people(dataframe: pd.DataFrame, test_size: float = 0.2):
    """
        Create train test split of DataFrame
        Args:
            :param dataframe: DataFrame to be split
            :param test_size: Percentage of total files to be split into test
        Return:
            :return X_train, X_test, y_train, y_test (tuple): Xs are list of
            df['language_num'] and Ys are df['native_language']
    """
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe['language_num'],
        dataframe['english_residence'], # english_residence, native_language
        test_size=test_size,
        train_size= 1 - test_size,
        random_state=1234
    )
    return x_train, x_test, y_train, y_test
