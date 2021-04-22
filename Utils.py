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
# - 9. Fourier_transform(audio): Fast Fourier transform (per frame)                               #
###################################################################################################


import os
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft, rfft


# Import the input source, audio(音频，np.array), sampleRate(采样率), duration(时长(s))
def get_input(audioPath):
    audio, sample_rate = sf.read(audioPath)
    if len(audio.shape) == 1:
        print("Mono audio.")
    else:
        print("Two channel or more.")
        audio1 = audio[:, 0]
        audio2 = audio[:, 1]
        audio = audio1 * 0.5 + audio2 * 0.5
    duration = len(audio) * 1.0 / sample_rate
    print("Input completed! Sample rate:", sample_rate, "| duration:", duration)
    return audio, sample_rate, duration


# De-pre-emphasis, to improve the total SNR (Signal to Noise Ratio), audio signal is pre-emphasised by default
# def de_pre_emphasis(audio, alpha):
#     for i in range(len(audio) - 1, 1, -1):
#         audio[i] += alpha * audio[i - 1]
#     audio[0] = audio[0] / (1 - alpha)
#     return audio


# Pre-emphasis, to improve the total SNR (Signal to Noise Ratio)
# def pre_emphasis(audio, alpha):
#     audio[0] = (1 - alpha) * audio[0]
#     for i in range(1, len(audio), 1):
#         audio[i] -= alpha * audio[i - 1]
#     return audio


# Self-Correlation
# Return the array and the basic frequency
def self_correlation(frame, maxT):
    print("Begin Self Correlation...")
    ret = []
    for l in range(0, maxT, 1):
        tmp_sum = 0
        for j in range(0, len(frame) - l, 1):
            tmp_sum += frame[j + l] * frame[j]
        ret.append(tmp_sum)
    print("Self Correlation completed, begin calculate the basic frequency...")
    # Get the peak of the curve, note that it is in FRAME unit!

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
    print("Build windows completed:", name)

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

    print("Divide frames completed! | frame amount", len(frames), "| frame size:", frameSize, "| frame shift:", frameShift, "| time for each frame:", round(time_for_each_frame * 1000, 3), "ms")
    return np.array(ori_frames), np.array(frames), time_for_each_frame


# Generate Short-Term Energy, which is the quadratic sum of sample points in one frame, return (np.array)
def generate_short_term_energy(frames):
    ret = []
    for i in range(0, len(frames), 1):
        energy = 0
        for j in range(0, len(frames[i]), 1):
            energy += pow(frames[i][j], 2)
        ret.append(energy)
    print("Generation of short term energy completed!")
    return np.array(ret)


# Calculate Zero-Crossing Rate (ZCR)
def cal_zero_crossing_rate(frames):
    ret = []
    for i in range(0, len(frames), 1):
        count = 0
        for j in range(0, len(frames[i]), 1):
            if frames[i][j] == 0 or (j > 0 and (
                    (frames[i][j - 1] > 0 and frames[i][j] < 0) or (frames[i][j - 1] < 0 and frames[i][j] > 0))):
                count += 1
        ret.append(count)
    return np.array(ret)


# Fast Fourier transform (per frame, Short-Time Frequency Spectrum )
def fourier_transform(frames, frameSize, sampleRate):
    fft_signals = []
    # X-axis transform in Fast Fourier Transform
    x = np.arange(int(frameSize / 2))
    for i in range(0, len(x), 1):
        x[i] = (sampleRate / frameSize) * x[i]
    # Short-Time Frequency Spectrum
    for i in range(0, len(frames), 1):
        fft_signal = []
        for j in range(0, len(frames[i]), 1):
            fft_signal.append(frames[i][j])
        if len(fft_signal) > 0:
            fft_signal = np.abs(fft(np.array(fft_signal)))
            # Cut half, and regularize the amplitude
            cut_fft_signal = []
            for j in range(0, int(len(fft_signal) / 2), 1):
                # Regularize the amplitude
                if j == 0:
                    cut_fft_signal.append(fft_signal[j] / len(fft_signal))
                else:
                    cut_fft_signal.append(fft_signal[j] / int(len(fft_signal / 2)))
            fft_signals.append(np.array(cut_fft_signal))
    return np.array(fft_signals), x


# Draw audio time domain diagram
def draw_time_domain_diagram(audio, energies, ori_frame, frame, ZCR, SCC, fft_signal, fft_x):
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
    plt.xlim(-100, 6000)
    # Original 30th frame
    plt.subplot(713)
    x = np.arange(len(ori_frame))
    plt.plot(x, ori_frame, 'black')
    plt.title("The 30th frame of original audio")
    plt.xlabel("sample points")
    plt.ylabel("Amplitude")
    # Windowed 30th frame
    plt.subplot(714)
    x = np.arange(len(frame))
    plt.plot(x, frame, 'black')
    plt.title("The 30th frame of windowed audio")
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

    # Self-Correlation
    plt.subplot(717)
    x = np.arange(len(SCC))
    plt.plot(x, SCC, 'black')
    plt.title("The Self-Correlation Curve of windowed audio")
    plt.xlabel("frame")
    plt.ylabel("Amplitude")

    plt.savefig("./output/output")
    plt.show()
