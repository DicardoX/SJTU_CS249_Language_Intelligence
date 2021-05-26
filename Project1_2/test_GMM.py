# test_GMM.py

import joblib
from matplotlib import pyplot as plt
import secrets
import time
from scipy import signal
import os
from evaluate import get_metrics
from sklearn import metrics

from Project1_2.Utils import *

# Secret generator
secret_generator = secrets.SystemRandom()

# Filter size
filter_size = 17
# Scale factor
scale_factor = 10000


# Test
def test_main():
    # Predicted result
    predicted_voice_curve = []

    print("---------------------------------------------------------------")
    print("Begin testing...")

    # Load model
    print("---------------------------------------------------------------")
    print("Reading model...")
    pos_GMM = joblib.load("./model/pos_model.pkl")
    neg_GMM = joblib.load("./model/neg_model.pkl")
    res = np.load("./model/model_distribution.npy", allow_pickle=True)
    means, delta_std = res[0], res[1]

    # Read features
    print("Reading features...")
    test_features_vector_list = np.load("./input/features/test_features.npy", allow_pickle=True)

    print("Amount of test features vector list:", len(test_features_vector_list))

    # Time unit for each frame, need to be synchronized with time_unit in data_construction.py
    time_unit = 30
    # Time shift
    time_shift = int(time_unit / 2)

    # Predict
    print("----------------------------------------")
    print("Predicting labels on test dataset...")
    for i in range(len(test_features_vector_list)):
        if i % 100 == 0:
            print("Iteration:", str(i), "/", str(len(test_features_vector_list)), "rounds")

        test_X = test_features_vector_list[i]

        # Normalization
        for j in range(len(test_X)):
            test_X[j] = np.subtract(test_X[j], means)
            test_X[j] = np.divide(test_X[j], delta_std)
            test_X[j] = np.multiply(test_X[j], scale_factor)

        pos_likelihood = list(pos_GMM.score_samples(test_X))
        neg_likelihood = list(neg_GMM.score_samples(test_X))

        # Model predict
        pred = [0 for j in range(len(pos_likelihood))]
        for j in range(0, len(pos_likelihood), 1):
            pred[j] = 0 if neg_likelihood[j] > pos_likelihood[j] else 1

        # Smooth the pred curve: Median filter
        pred = signal.medfilt(pred, kernel_size=filter_size)

        predicted_voice_curve.append(pred)

    # Write
    write_path = "./output/test_prediction.txt"
    print("----------------------------------------")
    print("Writing result into path: \"" + write_path + "\" ...")
    final_result = []
    wav_files = []
    # Get the names of wav file
    wav_files = os.listdir("../vad/wavs/test")
    # Sort the names of wav file, note that when test the accuracy of prediction on test dataset!!!
    wav_files.sort()

    print("----------------------------------------")
    print("Begin random human test...")
    # Randomly choose a wav file to perform human test
    test_idx = secret_generator.randint(0, len(wav_files) - 1)
    print("Wav file", test_idx, ":", wav_files[test_idx], "(Please search this file in ../vad/wavs/test,")
    print(" and make artificial judge between this wav figure and predicted curve)")        # Print file name
    print("----------------------------------------")
    plt.plot(predicted_voice_curve[test_idx])
    plt.savefig("./output/predict")
    plt.show()

    for i in range(len(predicted_voice_curve)):
        message_line = []
        wav_id = wav_files[i].replace(".wav", "")
        # Add wav id into message line
        message_line.append(wav_id)

        idx = 0
        while idx < len(predicted_voice_curve[i]):
            if predicted_voice_curve[i][idx] == 1:
                # Transform the predicted voice curve from frame-based into microsecond based
                begin_frame_idx = idx
                while idx < len(predicted_voice_curve[i]) - 1 and predicted_voice_curve[i][idx + 1] == 1:
                    idx += 1
                begin_moment = begin_frame_idx * time_shift
                end_moment = time_unit + idx * time_shift
                if begin_moment >= end_moment:
                    print("Error! begin_moment", str(begin_moment), "is no earlier than end_moment", str(end_moment))

                message_line.append(
                    str(round(float(begin_moment / 1000), 2)) + "," + str(round(float(end_moment / 1000), 2)))
            idx += 1
        # Transfer message line into correct format in string
        str_message_line = message_line[0]  # Wav id
        for j in range(1, len(message_line), 1):
            str_message_line = str_message_line + " " + message_line[j]
        final_result.append(str_message_line)

    # File operation
    if os.path.exists(write_path):
        os.remove(write_path)
    f = open(write_path, "w")
    for i in range(len(final_result)):
        f.writelines(final_result[i])
        f.write("\n")
    f.close()


if __name__ == '__main__':
    test_main()
