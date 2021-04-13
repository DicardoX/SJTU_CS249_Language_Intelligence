from Utils import *

frameSize = 2048
frameShift = int(frameSize / 2)
preEmphasisAlpha = 0.97
selfCorrelationCoefficient = int(frameSize / 500)

def main():
    # Input
    audio, sampleRate, duration = get_input("./input/input.wav")
    # Regularization
    audio = audio / np.max(audio)
    # Divide frames and add windows
    ori_frames, frames = divide_frames(audio, frameSize, frameShift)
    # Fourier transform (per second)
    fft_signals, fft_x = fourier_transform(frames, frameSize, sampleRate)
    # Calculate Short-term Energy
    energies = generate_short_term_energy(frames)
    # Calculate Zero-Crossing Rate
    ZCR = cal_zero_crossing_rate(frames)
    # Calculate Self-Correlation
    SCC = self_correlation(frames)
    print("Self Correlation completed!")
    # Draw results
    draw_time_domain_diagram(audio, energies, ori_frames[29], frames[29], ZCR, SCC, fft_signals[53], fft_x)


if __name__ == '__main__':
    main()