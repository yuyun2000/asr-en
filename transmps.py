'''
这个代码用来读取MP3并提取fbank特征
'''
import librosa

import scipy.io.wavfile as wavfile
import numpy as np
from prepro import compute_log_mel_fbank_fromsig
f,rate = librosa.load('./common_voice_en_12.mp3',sr=16000)
print(rate)
print(f.shape)
print(f.dtype)
f = f*32768
f = f.astype(np.int16)
out = compute_log_mel_fbank_fromsig(f,rate)
print(out.shape)

# sample_rate, signal = wavfile.read('t.wav')
# # print(signal.shape)
# # print(signal.dtype)
# # print(signal)
# out = compute_log_mel_fbank_fromsig(signal,sample_rate)
# print(out)