import secrets
from Project1.Utils import *
import warnings
import time

# Close warnings
warnings.filterwarnings("ignore")

# By default
frame_size = 512  # Width of window
frame_shift = int(frame_size / 2)  # Shift of window
sample_rate = 16000  # Sample rate

# Reconstructed dev dataset of features vector list
dev_features_vector_list_dataset = []
# Reconstructed test dataset of features vector list
test_features_vector_list_dataset = []
# Reformatted dev labels vector
labels_list = []

# Secret generator
secret_generator = secrets.SystemRandom()


# Update labels
def update_labels(frame_labels, start_moment, end_moment, time_unit):
    global frame_size, frame_shift, sample_rate
    # How to calculate teh start frame idx or end frame idx? Let t = frame_shift, k = frame_size
    # Answer: start point for each frame: 0, t, 2t, ..., nt
    #         end point for each frame: k, k+t, k+2t, ..., k+nt
    #         any point is in range [at, 1+bt], where b <= a
    #         this point covers frame b, frame b+1, ..., frame a
    t = time_unit * frame_shift / frame_size
    # Start moment
    start_frame_idx1 = int((start_moment * 1000 - time_unit) / t) + 1
    if start_moment * 1000 - time_unit < 0:
        start_frame_idx1 = 0
    start_frame_idx2 = int(start_moment * 1000 / t)
    # End moment
    end_frame_idx1 = int((end_moment * 1000 - time_unit) / t) + 1
    if end_moment * 1000 - time_unit < 0:
        end_frame_idx1 = 0
    end_frame_idx2 = int(end_moment * 1000 / t)

    # Update labels
    for i in range(len(frame_labels)):
        # As long as one frame is covered by voice pieces, it counts
        if start_frame_idx1 <= i <= end_frame_idx2:  # CAN BE OPTIMIZED!!!
            frame_labels[i] = 1


# Reformat label vector based on labels_list provided in construct_dataset()
# Change the label into FRAME unit: 0 for not voice, 1 for voice
def reformat_labels_list():
    global labels_list, frame_size, frame_shift, sample_rate
    global dev_features_vector_list_dataset
    tmp_labels_list = []

    print("Begin to reformat labels list...")
    print("----------------------------------------")

    # Time for each frame, in ms
    time_unit = int(frame_size * 1000 / sample_rate)

    # Time mark
    time_mark = time.process_time()

    # for i in range(10):
    for i in range(len(labels_list)):
        labels_vector = labels_list[i]
        # [0] for wav id, [1] for labels for each frame (a list)
        # tmp_labels_vector = [["wav id"], [0, 0, 0, 1, 1, ..., 0]]
        # Add wav id
        tmp_labels_vector = [str(labels_vector[0][0])]
        frame_labels = [0 for j in range(len(dev_features_vector_list_dataset[i]))]

        for j in range(1, len(labels_vector), 1):
            start_moment = float(labels_vector[j][0])
            end_moment = float(labels_vector[j][1])
            update_labels(frame_labels, start_moment, end_moment, time_unit)
        # print(len(frame_labels))
        # plt.plot(frame_labels)
        # plt.show()
        tmp_labels_vector.append(np.array(frame_labels))
        # print(tmp_labels_vector)
        tmp_labels_list.append(tmp_labels_vector)

        if i == 0:
            print("Estimated time for reformatting labels list:",
                  str(round(time.process_time() - time_mark, 3) * len(dev_features_vector_list_dataset)), "seconds...")

    # print(tmp_labels_list)
    # Update labels_list
    labels_list = tmp_labels_list.copy()


# Construct dev dataset: data_type 0 for dev, 1 for test, 2 for train
def construct_dataset(dataset_type):
    global frame_size, frame_shift, sample_rate
    global dev_features_vector_list_dataset, labels_list, test_features_vector_list_dataset

    if dataset_type == 0:
        print("----------------------------------------")
        print("Begin to construct dev dataset...")
        print("----------------------------------------")
        # Input
        audio_list, sample_rate_list, duration_list, labels_list = get_input("../vad", 0)  # 0 for dev dataset
    elif dataset_type == 1:
        print("----------------------------------------")
        print("Begin to construct test dataset...")
        print("----------------------------------------")
        # Input
        audio_list, sample_rate_list, duration_list = get_input("../vad", 1)  # 1 for test dataset
    else:
        print("----------------------------------------")
        print("Begin to construct train dataset...")
        print("----------------------------------------")
        # Input
        audio_list, sample_rate_list, duration_list, labels_list = get_input("../vad", 2)  # 2 for train dataset

    # Calculate frame size to make the unit time in 10 ~ 30ms, here is 25ms
    frame_size = int(30 * sample_rate_list[0] / 1000)
    frame_shift = int(frame_size / 2)
    # print("Frame size:", frame_size)

    # Total amount of samples
    total_amount = len(audio_list)

    # Regularization
    for i in range(total_amount):
        audio_list[i] = audio_list[i] / np.max(audio_list[i])

    # Construct the features vector list
    for i in range(10):
    # for i in range(total_amount):
        # Time mark
        if i == 0:
            time_mark = time.time()

        if i % 1 == 0:
            print("Iteration", str(i), "/", str(len(audio_list)), "for constructing features vector list...")
        audio = audio_list[i]
        duration = duration_list[i]
        sample_rate = sample_rate_list[i]

        # Divide frames and add windows
        ori_frames, frames, time_for_each_frame = divide_frames(audio, frame_size, frame_shift, duration)

        # Random frame order
        frame_order = secret_generator.randint(0, len(frames) - 1)
        # print("Random frame order:", frame_order)

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
        features_vector_list = construct_features_vector(frames=frames, STE_list=STE, ZCR_list=ZCR,
                                                         fft_max_arg_list=fft_max_arg,
                                                         mfcc_features_list=mfcc_features_list)

        # Push into dataset
        if dataset_type == 0:
            dev_features_vector_list_dataset.append(features_vector_list)
        elif dataset_type == 1:
            test_features_vector_list_dataset.append(features_vector_list)

        if i == 0:
            print("Estimated time for constructing features vector list dataset:",
                  str(round(time.time() - time_mark, 3) * len(audio_list)), "seconds...")

    print("----------------------------------------")


if __name__ == '__main__':
    # # Construct dev dataset, uncomment when reconstruction of dataset is needed
    # construct_dataset(0)
    # # Reformat labels list
    # reformat_labels_list()
    # print("Length of dev dataset:", len(dev_features_vector_list_dataset))
    # # Save data, uncomment when reconstruction of dataset is needed
    # print("Saving dev data...")
    # np.save("./input/features/dev_features.npy", dev_features_vector_list_dataset)
    # np.save("./input/labels/dev_labels.npy", labels_list)

    # Construct test dataset
    construct_dataset(1)
    print("Length of test dataset:", len(test_features_vector_list_dataset))
    # Save data, uncomment when reconstruction of dataset is needed
    print("Saving test data...")
    np.save("./input/features/test_features.npy", test_features_vector_list_dataset)

