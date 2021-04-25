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
# - 9. Construct_features_vector(frames, STE_list, ZCR_list, fft_max_arg_list,                    #
#       mfcc_features_list): Construct the features vector                                        #
###################################################################################################


import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
import time
from scipy import signal

# Detected frequency range
f_min = 1
f_max = 600

# Fundamental frequency range
f_human_min = 80
f_human_max = 500


# Import the input source, audio(音频，np.array), sampleRate(采样率), duration(时长(s))
# def get_input(audioPath):
#     audio, sample_rate = librosa.load(audioPath, sr=None, mono=True, offset=0.0)
#
#     duration = len(audio) * 1.0 / sample_rate
#     print("Input completed! Sample rate:", sample_rate, "| duration:", duration)
#     return audio, sample_rate, duration


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

    # duration = len(audio) * 1.0 / sample_rate
    # print("Input completed! Sample rate:", sample_rate, "| duration:", duration)
    # return audio, sample_rate, duration


# Self-Correlation
# Return the array and the basic frequency
def self_correlation(audio, frame_size, frame_shift, sample_rate):
    print("Begin Self Correlation...")

    # Time mark
    time_mark = time.process_time()

    f0_signal = librosa.yin(audio, fmin=f_min, fmax=f_max, sr=sample_rate, frame_length=len(audio), win_length=frame_size,
                            hop_length=frame_shift)

    # Smooth
    # Median filter
    f0_signal = signal.medfilt(f0_signal, 5)
    # Low pass
    f0_signal = signal.savgol_filter(f0_signal, 5, 2)

    # print("Calculation of fundamental frequency completed, totally " + str((round(time.process_time() - time_mark, 3))) + " seconds spent...")

    return np.array(f0_signal)

    # ret = []
    # for l in range(0, maxT, 1):
    #     tmp_sum = 0
    #     for j in range(0, len(frame) - l, 1):
    #         tmp_sum += frame[j + l] * frame[j]
    #     ret.append(tmp_sum)
    # print("Self Correlation completed, begin calculate the basic frequency...")
    # # Get the peak of the curve, note that it is in FRAME unit!

    # return ret


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
    # print("Build windows completed:", name)

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

    # # Remove the last two elms in frames, may result in mismatch between label and wav
    # frames.pop()
    # frames.pop()
    # ori_frames.pop()
    # ori_frames.pop()

    # print("Divide frames completed! | frame amount", len(frames), "| frame size:", frameSize, "| frame shift:",
    #       frameShift, "| time for each frame:", round(time_for_each_frame * 1000, 3), "ms")
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

    # Abandon the last frame
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
    # print("Calculation of MFCC completed!")
    return np.array(ret)


# Construct the features vector
def construct_features_vector(frames, STE_list, ZCR_list, fft_max_arg_list, mfcc_features_list):
    ret = []
    for i in range(len(frames)):
        features_vector = [STE_list[i], ZCR_list[i], fft_max_arg_list[i]]
        for j in range(len(mfcc_features_list[i][0])):
            features_vector.append(mfcc_features_list[i][0][j])
        ret.append(np.array(features_vector))
    return np.array(ret)


# Draw audio time domain diagram
def draw_time_domain_diagram(audio, energies, ori_frame, frame, ZCR, mfcc_features, fft_signal, fft_x):
    print("Begin draw results...")

    # Figure size
    plt.rcParams['figure.figsize'] = (20.0, 42.0)

    # Audio signal
    plt.subplot(711)  # row col pos
    x = np.arange(len(audio))
    plt.plot(x, audio, 'black')
    plt.title("Audio Signal on Time Domain")
    plt.xlabel("Sample points")
    plt.ylabel("Amplitude")
    # Fourier Transform
    plt.subplot(712)
    # X-axis transform in Fast Fourier Transform
    plt.plot(fft_x, fft_signal, 'black')
    plt.title("The Fourier Transform signal on Frequency Domain")
    plt.xlabel("frequency")
    plt.ylabel("Amplitude")
    ax = plt.gca()
    x_major_locator = plt.MultipleLocator(500)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, len(fft_x))
    # Original 30th frame
    plt.subplot(713)
    x = np.arange(len(ori_frame))
    plt.plot(x, ori_frame, 'black')
    plt.title("The frame of original audio")
    plt.xlabel("sample points")
    plt.ylabel("Amplitude")
    # Windowed 30th frame
    plt.subplot(714)
    x = np.arange(len(frame))
    plt.plot(x, frame, 'black')
    plt.title("The frame of windowed audio")
    plt.xlabel("sample points")
    plt.ylabel("Amplitude")
    # Short-Term Energy
    plt.subplot(715)
    x = np.arange(len(energies))
    plt.plot(x, energies, 'black')
    plt.title("Short-Term Energy")
    plt.xlabel("frame")
    plt.ylabel("Amplitude")
    # Zero-Crossing Rate
    plt.subplot(716)
    x = np.arange(len(ZCR))
    plt.plot(x, ZCR, 'black')
    plt.title("The Zero-Crossing Rate of windowed audio")
    plt.xlabel("frame")
    plt.ylabel("Amplitude")

    # MFCC
    plt.subplot(717)
    x = np.arange(len(mfcc_features))
    plt.plot(x, mfcc_features, 'black')
    plt.title("The MFCC Features of a Certain Frame")
    plt.xlabel("features")
    plt.ylabel("Amplitude")

    plt.savefig("./output/output")
    plt.show()
