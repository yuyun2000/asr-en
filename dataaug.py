import numpy as np
import cv2
import scipy.io.wavfile as wavfile
from audiomentations import *

augment = Compose([
    AddGaussianSNR(min_snr_in_db=3.0, max_snr_in_db=10.0, p=0.5),
    AirAbsorption( min_temperature = 10.0, max_temperature = 20.0,min_humidity = 30.0, max_humidity = 90.0,min_distance = 1.0, max_distance = 100.0,p=0.5),  #模拟空气低通
    FrequencyMask(min_frequency_band=0.0, max_frequency_band=0.5, p=0.5),
    Gain(min_gain_in_db=-6, max_gain_in_db=0, p=0.5),
    #BandPassFilter(  min_center_freq=4000, max_center_freq=4000.0, min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.99, \
    #               min_rolloff=12, max_rolloff=24, zero_phase=False,p=1.0,),  #must！
    BandPassFilter(  min_center_freq=200, max_center_freq=4000.0, min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.99, \
                   min_rolloff=12, max_rolloff=24, zero_phase=False,p=0.5,),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    PolarityInversion(p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    TanhDistortion(min_distortion = 0.01, max_distortion = 0.8, p = 0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
])

import librosa
signal, sample_rate = librosa.load('./103-1240-0006.flac', sr=None)

audsignal = augment(samples=signal,sample_rate=sample_rate)
# res=(res[0,:,0]*32767).astype(np.int16)


wavfile.write("103-1240-0006.wav", 16000, audsignal)
