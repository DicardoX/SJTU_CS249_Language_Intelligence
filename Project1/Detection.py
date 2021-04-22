from Utils import *
import secrets

frame_size = 1024                       # By default
frame_shift = int(frame_size / 2)
preEmphasisAlpha = 0.97
# self_correlation_coefficient = int(frame_size / 500)
max_T_frame = int(frame_size / 2)

# Secret generator
secret_generator = secrets.SystemRandom()


def main():
    # Input
    audio, sample_rate, duration = get_input("./input/input2.wav")
    # Regularization
    audio = audio / np.max(audio)
    # Divide frames and add windows
    ori_frames, frames, time_for_each_frame = divide_frames(audio, frame_size, frame_shift, duration)
    # Random frame order
    frame_order = secret_generator.randint(0, len(frames) - 1)
    print("Random frame order:", frame_order)

    # Fourier transform (per second)
    fft_signals, fft_x = fourier_transform(frames, frame_size, sample_rate)
    # Calculate Short-term Energy
    energies = generate_short_term_energy(frames)
    # Calculate Zero-Crossing Rate
    ZCR = cal_zero_crossing_rate(frames)
    # Calculate Self-Correlation
    SCC = self_correlation(frames[frame_order], max_T_frame)
    # Draw results
    draw_time_domain_diagram(audio, energies, ori_frames[frame_order], frames[frame_order], ZCR, SCC, fft_signals[53], fft_x)


if __name__ == '__main__':
    main()
