#This is Bartal Baranyi's and Bruno Milassin's work
#this topic is about music genre recognition using machine learning techniques. 
# The code will likely involve data preprocessing, feature extraction, and the 
# implementation of a machine learning model to classify music genres based on audio features.

#the code will likely contain elements from these libraries:
import librosa # audio loading and feature extraction
import numpy as np# numerical arrays
import pandas as pd # dataframes
#import scikitlearn # ML models, train/test split, metrics
import matplotlib.pyplot as plt # plots
import seaborn 
import os


"""Constants and parameters"""
DATASET_PATH = r"C:\01_BME\05_Intro to ML\00_MusicGenre\Data\genres_original"
N_MFCC = 20



def mfcc(file, n_mfcc=N_MFCC):
    """Extract MFCC features from an audio file."""

    try:
        audio, sample_rate = librosa.load(file)
    except Exception as e:
        print(f"Skipping {file} due to load error: {e}")
        return None

    mfcc_matrix = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc_matrix.T, axis=0)
    return mfcc_mean


def load_data(dataset_path=DATASET_PATH, n_mfcc=N_MFCC):
    """
    Load the dataset and extract features and labels.

    It goes through each genre folder, and for each audio file,
    it extracts the MFCC features, stores them in a list,
    and also stores the corresponding genre label in another list.
    """

    features = []
    labels = []

    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)

        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                file_path = os.path.join(genre_path, file)

                if file_path.endswith('.wav'):
                    mfcc_features = mfcc(file_path, n_mfcc=n_mfcc)

                    """ 
                    There was a loading error with at least one audio file in the dataset, 
                    to detect which file fails and continue processing, we catch exceptions
                    """
                    if mfcc_features is None:
                        continue

                    features.append(mfcc_features)
                    labels.append(genre)

    return np.array(features), np.array(labels)

