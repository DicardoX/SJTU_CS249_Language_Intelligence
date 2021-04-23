import secrets
from Project1.Utils import *

# By default
frame_size = 512  # 窗宽
frame_shift = int(frame_size / 2)  # 窗移
# preEmphasisAlpha = 0.97
max_T_frame = int(frame_size / 2)  # 最大检测频率（检测基频

# Secret generator
secret_generator = secrets.SystemRandom()


# Dev
def dev_main():
    global frame_size, frame_shift, max_T_frame

    # Input
    # audio, sample_rate, duration = get_input("./input/input.wav")
    audio_list, sample_rate_list, duration_list, labels_list = get_input("../vad", 0)           # 0 for dev dataset

    # Calculate frame size to make the unit time in 10 ~ 30ms, here is 25ms
    frame_size = int(30 * sample_rate_list[0] / 1000)
    frame_shift = int(frame_size / 2)
    max_T_frame = int(frame_size / 2)
    print("Frame size:", frame_size)

    # Regularization
    for i in range(len(audio_list)):
        audio_list[i] = audio_list[i] / np.max(audio_list[i])

    # Divide frames and add windows
    ori_frames, frames, time_for_each_frame = divide_frames(audio, frame_size, frame_shift, duration)
    # Random frame order
    frame_order = secret_generator.randint(0, len(frames) - 1)
    print("Random frame order:", frame_order)

    # Fourier transform
    # fft_signals, fft_x = fourier_transform(audio, frames, frame_size, frame_shift, sample_rate)
    fft_max_arg = fourier_transform(audio, frames, frame_size, frame_shift, sample_rate)
    # Calculate Short-term Energy
    STE = generate_short_term_energy(frames)
    # Calculate Zero-Crossing Rate
    ZCR = cal_zero_crossing_rate(frames, audio, frame_size)
    # # Calculate Self-Correlation
    # SCC = self_correlation(audio, frame_size, frame_shift, sample_rate)
    # Calculate MFCC
    mfcc_features_list = cal_MFCC(frames, frame_size, sample_rate)
    # # Draw results
    # draw_time_domain_diagram(audio, energies, ori_frames[frame_order], frames[frame_order], ZCR, mfcc_features_list[frame_order][0], fft_signals[frame_order],
    #                          fft_x)

    # Construct the features vector list
    features_vector_list = construct_features_vector(frames=frames, STE_list=STE, ZCR_list=ZCR, fft_max_arg_list=fft_max_arg, mfcc_features_list=mfcc_features_list)


def main():
    # Dev
    dev_main()


if __name__ == '__main__':
    main()
