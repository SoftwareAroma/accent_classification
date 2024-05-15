import librosa

@staticmethod
def extract_acoustic_features(
    file_path, 
    features=('mfcc', 'chroma_stft', 'spectral_centroid')
):
    """
        Extracts acoustic features from an audio file.
        Args:
            file_path (str): Path to the audio file.
            features (tuple, optional): A tuple of feature names to extract. Defaults to ('mfcc', 'chroma_stft', 'spectral_centroid').
        Returns:
            dict: A dictionary containing the extracted acoustic features, or None if an error occurs.
        Raises:
            ValueError: If an unsupported feature is requested.
    """

    try:
        y, sr = librosa.load(f'data/audio/{file_path}.wav')
        features_dict = {}
        for feature_name in features:
            if feature_name == 'mfcc':
                features_dict['mfcc'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            elif feature_name == 'chroma_stft':
                features_dict['chroma_stft'] = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            elif feature_name == 'spectral_centroid':
                features_dict['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
            elif feature_name == 'spectral_bandwidth':
                features_dict['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            elif feature_name == 'zero_crossing_rate':
                features_dict['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
                y=y, 
                frame_length=2048, 
                hop_length=512
            )
            elif feature_name == 'rmse':
                features_dict['rmse'] = librosa.feature.rms(y=y)
            else:
                raise ValueError(f"Unsupported feature: {feature_name}")

        return features_dict
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None  # Or handle error differently

@staticmethod
def extract_prosodic_features(
    file_path, 
    features=('pitch', 'intensity', 'duration')
):
    """
        Extracts prosodic features from an audio file.
        Args:
            file_path (str): Path to the audio file.
            features (tuple, optional): A tuple of feature names to extract. Defaults to ('pitch', 'intensity', 'duration').
        Returns:
            dict: A dictionary containing the extracted prosodic features.
        Raises:
            ValueError: If an unsupported feature is requested.
    """

    try:
        y, sr = librosa.load(f'data/audio/{file_path}.wav')
        prosodic_features = {}
        for feature_name in features:
            if feature_name == 'pitch':
                prosodic_features['pitch'] = librosa.yin(y=y, fmin=65, fmax=2093)
            elif feature_name == 'intensity':
                prosodic_features['intensity'] = librosa.feature.rms(y=y)
            elif feature_name == 'duration':
                prosodic_features['duration'] = len(y) / sr
            elif feature_name == 'formants':
                # Implement formant extraction using librosa or a custom solution
                prosodic_features['formants'] = None  # Placeholder for now
            else:
                raise ValueError(f"Unsupported feature: {feature_name}")
        return prosodic_features
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None  # Handle loading errors

# extract plp
@staticmethod 
def extract_plp(file_path):
    y, sr = librosa.load(f'data/audio/{file_path}.wav')
    plp = librosa.beat.plp(y=y, sr=sr)
    return plp