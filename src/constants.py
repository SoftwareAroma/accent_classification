# !/usr/bin/env python
"""
    All non changing values of the model can be found here
    There are some that can be modified to suit the need or
    outcome of the model, like learning rate, number of epochs 
    and others. 
"""
OUTPUT_DIR: str = "output/"
CSV_FILE_PATH: str = "bio_metadata.csv"
DATASET_DIR: str = "data/"
RAW_DATA_DIR: str = "data/raw/"
AUDIO_DATA_DIR: str = "audio/"
AUDIO_FILE_PATH: str = "data/audio/{}.wav"
SEGMENTED_DATA_DIR: str = ""
SILENCE_THRESHOLD: float = .01
RATE: int = 24000
N_MFCC: int = 13
COL_SIZE: int = 30
EPOCHS: int = 50  #10 35 #250
LEARNING_RATE = 0.001
LANGUAGE_LIST: list[str] = [
    'english',
    'arabic',
    'mandarin',
    'dutch',
    'french',
    'german',
    'italian',
    'korean',
    'portuguese',
    'ga',
]

ROOT_URL: str = "http://accent.gmu.edu/"
BROWSE_LANGUAGE_URL: str = 'browse_language.php?function=find&language={}'
AUDIO_URL: str = 'http://accent.gmu.edu/soundtracks/{}.mp3'
AUDIO_URL_TEST: str = 'http://accent.gmu.edu/soundtracks/ewe1.mp3'
WAIT: float = 1.2
DEBUG: bool = True
