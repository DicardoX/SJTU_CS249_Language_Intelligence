###################################################################################################
#                                      UTILS LIST                                                 #
###################################################################################################
# - 1. Get_input(audioPath): Get audio input                                                      #
# - 2. Build_windows(name, N): Build window on audio wave                                         #
# - 3. Divide_frames(audio, frameSize): Divide Frame, return a list of frames (np.array)          #
# - 4. Draw_time_domain_diagram(audio): Draw the results                                          #
# - 5. Generate_short_term_energy(frames): Generate Short-Term Energy, which is the quadratic sum #
#      of sample points in one frame, return (np.array)                                           #
# - 6. Cal_zero_crossing_rate(frames): Calculate Zero-Crossing Rate (ZCR)                         #
# - 7. De_pre_emphasis(audio, alpha): to improve the total SNR (Signal to Noise Ratio), audio     #
#      signal is pre-emphasised by default                                                        #
# - 8. Pre_emphasis(audio, alpha): to improve the total SNR (Signal to Noise Ratio)               #
###################################################################################################


import os
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft, rfft

# Import the input source, audio(音频，np.array), sampleRate(采样率), duration(时长(s))
def get_input(audioPath):
    audio, sampleRate = sf.read(audioPath)
    if len(audio.shape) == 1:
        print("Mono audio.")
    else:
        print("Two channel or more.")
        audio1 = audio[:, 0]
        audio2 = audio[:, 1]
        audio = audio1 * 0.5 + audio2 * 0.5
    duration = len(audio) * 1.0 / sampleRate
    return audio, sampleRate, duration

# De-pre-emphasis, to improve the total SNR (Signal to Noise Ratio), audio signal is pre-emphasised by default
def de_pre_emphasis(audio, alpha):
    for i in range(len(audio) - 1, 1, -1):
        audio[i] += alpha * audio[i-1]
    audio[0] = audio[0] / (1 - alpha)
    # for i in range(1, len(audio), 1):
    #     audio[i] += alpha * audio[i - 1]
    return audio

# Pre-emphasis, to improve the total SNR (Signal to Noise Ratio)
def pre_emphasis(audio, alpha):
    audio[0] = (1 - alpha) * audio[0]
    for i in range(1, len(audio), 1):
        audio[i] -= alpha * audio[i - 1]
    return audio

# Self-Correlation
def self_correlation(frames, k):
    ret = []
    for i in range(0, len(frames), 1):
        tmpSum = 0
        for j in range(0, len(frames[i]) - k, 1):
            tmpSum += frames[i][j] * frames[i][j + k]
        ret.append(tmpSum)
    return ret

# Implement Window: Rectangle/Hanning/Hamming, N is window length
def build_windows(name='Hamming', N=20):
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
def divide_frames(audio, frameSize, frameShift):
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
    return np.array(ori_frames), np.array(frames)

# Generate Short-Term Energy, which is the quadratic sum of sample points in one frame, return (np.array)
def generate_short_term_energy(frames):
    ret = []
    for i in range(0, len(frames), 1):
        energy = 0
        for j in range(0, len(frames[i]), 1):
            energy += pow(frames[i][j], 2)
        ret.append(energy)
    return np.array(ret)

# Calculate Zero-Crossing Rate (ZCR)
def cal_zero_crossing_rate(frames):
    ret = []
    for i in range(0, len(frames), 1):
        count = 0
        for j in range(0, len(frames[i]), 1):
            if frames[i][j] == 0 or (j > 0 and ((frames[i][j - 1] > 0 and frames[i][j] < 0) or (frames[i][j - 1] < 0 and frames[i][j] > 0))):
                count += 1
        ret.append(count)
    return np.array(ret)

# Draw audio time domain diagram
def draw_time_domain_diagram(audio, energys, ori_frame, frame, ZCR, SCC):
    # Figure size
    plt.rcParams['figure.figsize'] = (20.0, 36.0)

    # Audio signal
    plt.subplot(611)  # row col pos
    plt.plot(audio)
    plt.title("Audio Signal on Time Domain")
    plt.xlabel("Sample points")
    plt.ylabel("Amplitude")
    # Original 30th frame
    plt.subplot(612)
    plt.plot(ori_frame)
    plt.title("The 30th frame of original audio")
    plt.xlabel("sample points")
    plt.ylabel("Amplitude")
    # Windowed 30th frame
    plt.subplot(613)
    plt.plot(frame)
    plt.title("The 30th frame of windowed audio")
    plt.xlabel("sample points")
    plt.ylabel("Amplitude")
    # Short-Term Energy
    plt.subplot(614)
    plt.plot(energys)
    plt.title("Short-Term Energy")
    plt.xlabel("frame")
    plt.ylabel("Amplitude")
    # Zero-Crossing Rate
    plt.subplot(615)
    plt.plot(ZCR)
    plt.title("The Zero-Crossing Rate of windowed audio")
    plt.xlabel("frame")
    plt.ylabel("Amplitude")
    # Self-Correlation
    plt.subplot(616)
    plt.plot(SCC)
    plt.title("The Self-Correlation of windowed audio")
    plt.xlabel("frame")
    plt.ylabel("Amplitude")

    plt.savefig("./output/output")
    plt.show()

