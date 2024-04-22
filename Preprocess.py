import numpy as np

def normalize_frames(signal, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in signal])

import python_speech_features as psf

def extract_features(waveform, sample_rate):
    filter_banks,energies = psf.fbank(waveform, samplerate = sample_rate,numcep=64,nfilt=64,winlen=0.025, winstep=0.01)
    mfcc = normalize_frames(signal=filter_banks)
    
    return np.array(mfcc, dtype = np.float32)

import librosa

def read_audio(filename,sample_rate = 16000):
    audio, _ = librosa.load(filename, sr = sample_rate, mono=True)
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    audio = audio[offsets[0]:offsets[-1]]
    mfcc = extract_features(audio,sample_rate)
    return mfcc

from random import choice 

def pad_mfcc(mfcc,max_length):
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc,np.tile(np.zeros(mfcc.shape[1]),(max_length - len(mfcc),1))))
    return mfcc

def trim(mfcc,max_length=160):
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r:r + max_length]
    else:
        s = pad_mfcc(mfcc,max_length)
    return np.expand_dims(s,axis=-1)

def batch_cosine_similarity(x1, x2):
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul, axis=1)
    
    return s

DEFAULT_SAMPLE_RATE = 16000

def extract_mfcc(clip, nr_mfcc):
    # downsample all clips to 16kHz
    signal, sr = librosa.load(clip, duration=3, sr=DEFAULT_SAMPLE_RATE)
    mfcc_feature = librosa.feature.mfcc(y= signal, n_mfcc=nr_mfcc, sr=sr, hop_length=256)
    delta_feature = librosa.feature.delta(mfcc_feature)
    mfcc_feature = np.mean(mfcc_feature.T, axis=0)
    delta_feature = np.mean(delta_feature.T, axis=0)

    return mfcc_feature, delta_feature

def extract_lpc(clip, nr_mfcc):
    # downsample all clips to 16kHz
    signal, sr = librosa.load(clip, sr=DEFAULT_SAMPLE_RATE)
    lpc_feature = librosa.lpc(y=signal, order=nr_mfcc-1)

    return lpc_feature

import librosa
import scipy.signal as sps
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
import soundfile as sf

import scipy.signal as sps

def download_sample(signal, sr, new_sr):
    # Resample data
    number_of_samples = round(len(signal) * float(new_sr) / sr)
    signal_resampled = sps.resample(signal, number_of_samples)
    return signal_resampled

def zero_crossing_rate(clip, splits, target_sr):
    # Read machine sound
    s, fs = sf.read(clip)
    
    # Resample data if necessary
    if fs != target_sr:
        print(f'Resampling clip: {clip} with rate: {fs} to target rate: {target_sr}')
        s = download_sample(s, fs, target_sr)
    
    duration = len(s) / float(target_sr)
    window = duration / splits
    
    # Calculate samples per split
    samples_per_split = int(target_sr * window)
    
    # Initialize array to store ZCR for each split
    zcr_per_split = []
    
    # Calculate ZCR for each split
    for i in range(splits):
        # Get the start and end indices for the current split
        start_idx = i * samples_per_split
        end_idx = start_idx + samples_per_split
        
        # Extract the signal for the current split
        split_signal = s[start_idx:end_idx]
        
        # Calculate ZCR for the current split
        zcr = calculate_zcr(split_signal)
        
        # Store the ZCR value
        zcr_per_split.append(zcr)
    
    return zcr_per_split

def calculate_zcr(signal):
    # Calculate the number of zero crossings
    zcr = np.mean(np.abs(np.diff(np.sign(signal))) > 0)
    return zcr


def feature(path, n_mfcc = 13):
    sound_feature = []
    mfcc_feature,delta_feature = extract_mfcc(path,n_mfcc)
    lpc = extract_lpc(path,n_mfcc)
    zcr = zero_crossing_rate(path,n_mfcc,16000)

    sound_feature.append(np.hstack([mfcc_feature,delta_feature,lpc,zcr]))
    
    return sound_feature


