###################################################################################################
#                                      UTILS LIST                                                 #
###################################################################################################
# - 1. Get_input(audioPath): Get audio input                                                      #
# - 2. Build_windows(name, N): Build window on audio wave                                         #
# - 3. Divide_frames(audio, frameSize): Divide Frame, return a list of frames (np.array)          #
# - 4. Draw_time_domain_diagram(audio): Draw the results                                          #
# - 5. Generate_short_term_energy(frames): Generate Short-Term Energy, which is the quadratic sum #
#       of sample points in one frame, return (np.array)                                          #
# - 6. Cal_zero_crossing_rate(frames): Calculate Zero-Crossing Rate (ZCR)                         #
# - 7. Fourier_transform(audio): Fast Fourier transform (per frame)                               #
# - 8. Cal_MFCC(frames, frame_size, sample_rate): MFCC (Mel-frequency Cepstral Coefficients)      #
# - 9. Perceptual_linear_predictive(frames, frame_size, frame_shift, sample_rate): Calculate PLP  #
#      (perceptual linear predictive) coefficients (top 15)                                       #
# - 10. Construct_features_vector(frames, STE_list, ZCR_list, fft_max_arg_list,                   #
#      mfcc_features_list): Construct the features vector                                         #
###################################################################################################


import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
import time
from scipy import signal

# Fundamental frequency range
f_human_min = 80
f_human_max = 500


# Get the dataset
# Dataset_type: 0 for dev, 1 for test, 2 for train
def get_input(dirPath, dataset_type):
    # File pre-definition
    wav_files = []
    label_file = ""
    # Get files name list
    if dataset_type == 0:
        # dev
        wav_dirPath = dirPath + "/wavs/dev"
        wav_files = os.listdir(wav_dirPath)
        label_file = dirPath + "/data/dev_label.txt"
    elif dataset_type == 1:
        # test
        wav_dirPath = dirPath + "/wavs/test"
        wav_files = os.listdir(wav_dirPath)
    else:
        # train
        wav_dirPath = dirPath + "/wavs/train"
        wav_files = os.listdir(wav_dirPath)
        label_file = dirPath + "/data/train_label.txt"

    # List container
    audio_list = []
    sample_rate_list = []
    duration_list = []

    # Sort the file
    wav_files.sort()

    # Traverse the wave
    for i in range(len(wav_files)):
        audio_path = wav_dirPath + "/" + wav_files[i]
        audio, sample_rate = librosa.load(audio_path, sr=None, mono=True, offset=0.0)
        audio_list.append(audio)
        sample_rate_list.append(sample_rate)
        duration_list.append(len(audio) * 1.0 / sample_rate)

    # Traverse the label
    lines = []
    labels_list = []
    if dataset_type == 0 or dataset_type == 2:
        for line in open(label_file):
            lines.append(line)
        # Sort labels based on wav_id
        lines.sort()
        # Construct the labels_list
        for i in range(len(lines)):
            # Split labels
            label_message = lines[i].split(" ")
            label_message_list = []
            for j in range(len(label_message)):
                # Remove \n
                label_message[j] = label_message[j].replace("\n", "")
                # Split begin moment and end moment
                label_message_list.append(np.array(label_message[j].split(",")))
            labels_list.append(np.array(label_message_list))
        labels_list = np.array(labels_list)

        # Check correctness in order
        is_inOrder = True
        for i in range(len(wav_files)):
            if wav_files[i] != (str(labels_list[i][0][0]) + ".wav"):
                print("Error occurred when comparing the order match of name between wav files and label...")
                is_inOrder = False
                break
        if is_inOrder:
            print("Successfully check the the order match of name between wav files and label!")
        print("----------------------------------------")
        return np.array(audio_list), np.array(sample_rate_list), np.array(duration_list), np.array(labels_list)
    else:
        return np.array(audio_list), np.array(sample_rate_list), np.array(duration_list)


# Implement Window: Rectangle/Hanning/Hamming, N is window length
def build_windows(name='Hamming', N=1024):
    # Default none
    window = None
    # Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    # Hanning
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    # Rectangle
    elif name == 'Rectangle':
        window = np.ones(N)

    return window


# Divide Frame, return a list of frames (np.array)
def divide_frames(audio, frameSize, frameShift, duration):
    frames = []
    ori_frames = []
    # Build windows
    windows = build_windows('Hanning', frameSize)
    for i in range(0, len(audio), frameShift):
        frame = []
        ori_frame = []
        for j in range(i, min(i + frameSize, len(audio)), 1):
            ori_frame.append(audio[j])
            # Add window
            frame.append(audio[j] * windows[frameSize - (j - i) - 1])
        ori_frames.append(np.array(ori_frame))
        frames.append(np.array(frame))
    time_for_each_frame = duration / (len(frames) - (len(frames) - 1) * (frameShift / frameSize))

    return np.array(ori_frames), np.array(frames), time_for_each_frame


# Generate Short-Term Energy, which is the quadratic sum of sample points in one frame, return (np.array)
def generate_short_term_energy(frames):
    ret = []
    for i in range(0, len(frames), 1):
        energy = 0
        for j in range(0, len(frames[i]), 1):
            energy += pow(frames[i][j], 2)
        ret.append(energy)
    # print("Generation of short term energy completed!")
    return np.array(ret)


# Calculate Zero-Crossing Rate (ZCR)
def cal_zero_crossing_rate(frames, audio, frame_size):
    ret = librosa.feature.zero_crossing_rate(audio, frame_size, int(frame_size / 2))
    # print("Calculation of zero crossing rate completed!")
    # The shape of ret is (1, frame_amount)
    return ret[0]


# Fast Fourier transform (per frame, Short-Time Frequency Spectrum )
def fourier_transform(audio, frames, frame_size, frame_shift, sampleRate):
    spectrum = []
    ret = []
    for i in range(len(frames)):
        spectrum.append(np.abs(
            librosa.stft(frames[i], n_fft=frame_size, hop_length=frame_size + 1, win_length=frame_size, window="hann")))

    for i in range(len(frames)):
        # In x-axis, the real frequency = i * (sample rate / # of points in this frame)
        # the upper bound of the loop is int(sampleRate / 2) + 1
        tmpList = [0 for i in range(int(sampleRate / 2) + 1)]
        for j in range(len(spectrum[i])):
            tmpList[int(j * (sampleRate / frame_size))] = (spectrum[i][j][0])
        ret.append(np.argmax(np.array(tmpList)))
    return ret
    # return ret, np.arange(len(ret[0]))


# MFCC (Mel-frequency Cepstral Coefficients)
def cal_MFCC(frames, frame_size, sample_rate):
    ret = []
    for i in range(len(frames)):
        # 39 Dimensions MFCC
        mfcc_features = librosa.feature.mfcc(frames[i], sr=sample_rate, S=None, n_mfcc=13, hop_length=frame_size + 1, dct_type=2, norm='ortho')
        mfcc_features = mfcc_features.reshape([1, -1])
        ret.append(mfcc_features)
    return np.array(ret)


# Calculate PLP (perceptual linear predictive) coefficients (top 15)
def perceptual_linear_predictive(frames, frame_size, frame_shift, sample_rate):
    ret = []

    # Transform the linear frequency coordinate to Bark coordinate
    def bark_transform(x):
        return 6 * np.log10(x / (1200 * np.pi) + ((x / (1200 * np.pi)) ** 2 + 1) ** 0.5)

    # Equal loudness curve
    def equal_loudness(x):
        return ((x ** 2 + 56.8e6) * x ** 4) / ((x ** 2 + 6.3e6) ** 2 * (x ** 2 + 3.8e8))

    for idx in range(len(frames)):
        # FFT
        a = np.fft.fft(frames[idx])
        # Square, and only half
        N = int(frame_size / 2)
        b = np.square(abs(a[0:N]))
        # 频率分辨率
        df = sample_rate / N
        # 只取大于0部分的频率
        i = np.arange(N)
        # 得到实际频率坐标
        freq_hz = i * df
        # plt.plot(freq_hz, b)  # 得到该帧信号的功率谱
        # plt.show()
        freq_w = 2 * np.pi * np.array(freq_hz)  # 转换为角频率
        freq_bark = bark_transform(freq_w)  # 再转换为bark频率
        point_hz = [250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400]
        # 选取的临界频带数量一般要大于10，覆盖常用频率范围，这里我选取了15个中心频点
        point_w = 2 * np.pi * np.array(point_hz)  # 转换为角频率
        point_bark = bark_transform(point_w)  # 转换为bark频率
        bank = np.zeros((15, len(b)))  # 构造15行(frame_size / 2)列的矩阵，每一行为一个滤波器向量
        filter_data = np.zeros(15)  # 构造15维频带能量向量

        for j in range(15):
            for k in range(len(b)):
                omg = freq_bark[k] - point_bark[j]
                if -1.3 < omg < -0.5:
                    bank[j, k] = 10 ** (2.5 * (omg + 0.5))
                elif -0.5 < omg < 0.5:
                    bank[j, k] = 1
                elif 0.5 < omg < 2.5:
                    bank[j, k] = 10 ** (-1.0 * (omg - 0.5))
                else:
                    bank[j, k] = 0
            filter_data[j] = np.sum(b * bank[j])  # 滤波后将该段信号相加，最终得到15维的频带能量

        equal_data = equal_loudness(point_w) * filter_data
        cubic_data = equal_data ** 0.33
        # 做30点的ifft，得到30维PLP向量
        plp_data = np.fft.ifft(cubic_data, 30)
        # print(plp_data)
        # print(len(plp_data))
        # features = librosa.lpc(abs(plp_data), 15)

        # 取前15维作语音信号处理
        features = plp_data[0:15]
        # PLP will cause complex number
        ret.append(abs(features))

    return ret


# Construct the features vector
def construct_features_vector(frames, STE_list, ZCR_list, plp_features_list, mfcc_features_list):
    ret = []
    for i in range(len(frames)):
        # STE, ZCR
        features_vector = [STE_list[i], ZCR_list[i]]
        # PLP
        for j in range(len(plp_features_list[i])):
            features_vector.append(plp_features_list[i][j])
        # MFCC
        for j in range(len(mfcc_features_list[i][0])):
            features_vector.append(mfcc_features_list[i][0][j])
        ret.append(np.array(features_vector))
    return np.array(ret)


# # Self-Correlation
# # Return the array and the basic frequency
# def self_correlation(audio, frame_size, frame_shift, sample_rate):
#     print("Begin Self Correlation...")
#
#     # Time mark
#     time_mark = time.process_time()
#
#     f0_signal = librosa.yin(audio, fmin=f_min, fmax=f_max, sr=sample_rate, frame_length=len(audio), win_length=frame_size,
#                             hop_length=frame_shift)
#
#     # Smooth
#     # Median filter
#     f0_signal = signal.medfilt(f0_signal, 5)
#     # Low pass
#     f0_signal = signal.savgol_filter(f0_signal, 5, 2)
#
#     return np.array(f0_signal)


# # Draw audio time domain diagram
# def draw_time_domain_diagram(audio, energies, ori_frame, frame, ZCR, mfcc_features, fft_signal, fft_x):
#     print("Begin draw results...")
#
#     # Figure size
#     plt.rcParams['figure.figsize'] = (20.0, 42.0)
#
#     # Audio signal
#     plt.subplot(711)  # row col pos
#     x = np.arange(len(audio))
#     plt.plot(x, audio, 'black')
#     plt.title("Audio Signal on Time Domain")
#     plt.xlabel("Sample points")
#     plt.ylabel("Amplitude")
#     # Fourier Transform
#     plt.subplot(712)
#     # X-axis transform in Fast Fourier Transform
#     plt.plot(fft_x, fft_signal, 'black')
#     plt.title("The Fourier Transform signal on Frequency Domain")
#     plt.xlabel("frequency")
#     plt.ylabel("Amplitude")
#     ax = plt.gca()
#     x_major_locator = plt.MultipleLocator(500)
#     ax.xaxis.set_major_locator(x_major_locator)
#     plt.xlim(0, len(fft_x))
#     # Original 30th frame
#     plt.subplot(713)
#     x = np.arange(len(ori_frame))
#     plt.plot(x, ori_frame, 'black')
#     plt.title("The frame of original audio")
#     plt.xlabel("sample points")
#     plt.ylabel("Amplitude")
#     # Windowed 30th frame
#     plt.subplot(714)
#     x = np.arange(len(frame))
#     plt.plot(x, frame, 'black')
#     plt.title("The frame of windowed audio")
#     plt.xlabel("sample points")
#     plt.ylabel("Amplitude")
#     # Short-Term Energy
#     plt.subplot(715)
#     x = np.arange(len(energies))
#     plt.plot(x, energies, 'black')
#     plt.title("Short-Term Energy")
#     plt.xlabel("frame")
#     plt.ylabel("Amplitude")
#     # Zero-Crossing Rate
#     plt.subplot(716)
#     x = np.arange(len(ZCR))
#     plt.plot(x, ZCR, 'black')
#     plt.title("The Zero-Crossing Rate of windowed audio")
#     plt.xlabel("frame")
#     plt.ylabel("Amplitude")
#
#     # MFCC
#     plt.subplot(717)
#     x = np.arange(len(mfcc_features))
#     plt.plot(x, mfcc_features, 'black')
#     plt.title("The MFCC Features of a Certain Frame")
#     plt.xlabel("features")
#     plt.ylabel("Amplitude")
#
#     plt.savefig("./output/output")
#     plt.show()
