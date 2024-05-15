# !/usr/bin/env python

"""
    The GetAudio class is used to download audio files from the web
    and save them to a specified folder
"""
import os
import time
import librosa 
import pandas as pd
import soundfile as sf
import urllib.request
from pydub import AudioSegment

from constants import CSV_FILE_PATH, AUDIO_DATA_DIR, AUDIO_URL, DEBUG

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class GetAudio:
    def __init__(
            self,
            csv_filepath: str,
            destination_folder: str = AUDIO_DATA_DIR,
            wait: float = 1.5,
            debug: bool = DEBUG
    ):
        """
            Initializes GetAudio class object
            Args:
                :param destination_folder: Folder where audio files will be saved
                :param wait : Length (in seconds) between web requests
                :param debug: Outputs status indicators to console when True
        """
        self.csv_filepath = csv_filepath
        self.audio_df = pd.read_csv(csv_filepath)
        self.url = AUDIO_URL
        self.destination_folder = os.path.join("data", destination_folder) 
        self.wait = wait
        self.debug = debug

    def check_path(self):
        """
            Checks if self.destination_folder exists.
            If not, a folder called self.destination_folder is created
        """
        folder_path = os.path.join("data", self.destination_folder)  # Combine data and destination paths
        if not os.path.exists(folder_path):
            if self.debug:
                print('{} does not exist, creating'.format(self.destination_folder))
            os.makedirs(folder_path)

    def get_audio(self):
        '''
            Retrieves all audio files from 'language_num' column of self.audio_df
            If audio file already exists, move on to the next
            :return (int): Number of audio files downloaded
        '''
        self.check_path()
        counter = 0
        for lang_num in self.audio_df['language_num']:
            if os.path.exists(self.destination_folder +'{}.wav'.format(lang_num)):
                if self.debug:
                    print('File Already here {}'.format(lang_num))
                continue
            if not os.path.exists(self.destination_folder +'{}.wav'.format(lang_num)):
                if self.debug:
                    print('downloading {}'.format(lang_num))
                (filename, headers) = urllib.request.urlretrieve(self.url.format(lang_num))
                sound = AudioSegment.from_mp3(filename)
                sound.export(
                    self.destination_folder + "{}.wav".format(lang_num), format="wav"
                )
                counter += 1
        return counter


def get_main_audio():
    """
        Example console command
        python GetAudio.py audio_metadata.csv
    """
    csv_file = CSV_FILE_PATH
    # read the csv file
    # df = pd.read_csv(csv_file)
    # print(df.head())
    ga = GetAudio(csv_filepath=csv_file)
    ga.get_audio()
