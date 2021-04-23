# Test.py

import librosa
import numpy as np
from matplotlib import pyplot as plt

N = 4096

# 假设为1s采样4096次

cos_signal = np.array([(np.cos(2 * 500 * np.pi * n / (N - 1)) + np.cos(2 * 200 * np.pi * n / (N - 1))) for n in range(N)])
# cos_signal = np.array([(np.cos(2 * 500 * np.pi * n / (N - 1))) for n in range(N)])


# cos_signal = np.abs(librosa.stft(cos_signal, n_fft=1024, hop_length=512, win_length=None, window="hann"))

# f0_signal = librosa.pyin(cos_signal, fmin=1, fmax=1000, sr=4096, frame_length=4096, win_length=1024, hop_length=1024 + 1)

# ret = librosa.feature.zero_crossing_rate(cos_signal, 1024, 512)

ret = librosa.feature.mfcc(cos_signal, sr=4096, S=None, n_mfcc=13, hop_length=4097, dct_type=2, norm='ortho')

print(ret.shape)

ret = ret.reshape([1, 13])


# ret = [[0 for i in range(int(4096 / 2) + 1)] for j in range(len(cos_signal[0]))]
# print(np.array(ret).shape)
# for i in range(len(cos_signal[0])):
#     for j in range(len(cos_signal)):
#         ret[i][int(j * 4096 / 1024)] = cos_signal[j][i]
#
#
# print(np.array(ret).shape)
#
#
# print(cos_signal.shape)

# plt.xlim(0, 2 * np.pi)

plt.plot(ret[0])

plt.show()


