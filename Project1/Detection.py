from Project1.Utils import *
import warnings
import sklearn
import joblib
from matplotlib import pyplot as plt
import secrets
import time
from scipy import signal
import os
from evaluate import get_metrics
from sklearn import metrics

# Close warnings
# warnings.filterwarnings("ignore")

# Reconstructed dev dataset of features vector list
dev_features_vector_list_dataset = []
# Reconstructed test dataset of features vector list
test_features_vector_list_dataset = []
# Reconstructed train dataset of features vector list
train_features_vector_list_dataset = []
# Reformatted dev labels vector
labels_list = []

# Secret generator
secret_generator = secrets.SystemRandom()

# Solver for Logistic Regression
solver_name = "liblinear"

# Logistic Regression Model
# Solver="liblinear" is a good choice for small dataset
model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.001, random_state=123, solver=solver_name)

# Filter size
filter_size = 17
# Threshold
voice_threshold = 0.83


# Generate batch, put all frames together into a batch
def generate_batch():
    global dev_features_vector_list_dataset, labels_list  # , batch_size
    batch = []
    label = []

    for i in range(len(dev_features_vector_list_dataset)):
        for j in range(len(dev_features_vector_list_dataset[i])):
            batch.append(dev_features_vector_list_dataset[i][j])
            label.append(labels_list[i][1][j])
    return batch, label


# Evaluate by using get_metrics() in evaluate.py
def evaluate_dev():
    global dev_features_vector_list_dataset, labels_list
    global model
    print("----------------------------------------")
    print("Evaluating labels on dev dataset by using get_metrics() in evaluate.py...")

    # Predict
    auc_score = 0
    eer_score = 0
    for i in range(len(dev_features_vector_list_dataset)):
        if i % 100 == 0 and i != 0:
            print("Iteration:", str(i), "/", str(len(dev_features_vector_list_dataset)),
                  "rounds | Current average AUC:", float(auc_score / i), "| Current average EER:", float(eer_score / i))
        voice_prob = model.predict_proba(dev_features_vector_list_dataset[i])
        pred_Y = []
        for j in range(len(voice_prob)):
            pred_Y.append(voice_prob[j][1])

        # Smooth the predicted curve: Median filter
        pred_Y = signal.medfilt(pred_Y, kernel_size=filter_size)
        # Normalized the predicted voice curve into 0 / 1 array
        for j in range(len(pred_Y)):
            pred_Y[j] = 1 if pred_Y[j] > voice_threshold else 0

        label_Y = labels_list[i][1]
        cur_auc, cur_eer = get_metrics(pred_Y, label_Y)
        auc_score += cur_auc
        eer_score += cur_eer

        # Plot the ROC curve of the first and last wav file
        if i == 0:
            metrics.plot_roc_curve(model, dev_features_vector_list_dataset[i], label_Y)
            plt.savefig("./output/roc_curve_1")
            plt.show()
        if i == len(dev_features_vector_list_dataset) - 1:
            metrics.plot_roc_curve(model, dev_features_vector_list_dataset[i], label_Y)
            plt.savefig("./output/roc_curve_2")
            plt.show()

    return float(auc_score / len(dev_features_vector_list_dataset)), float(eer_score / len(dev_features_vector_list_dataset))


# Train model
def model_train(dev_X, dev_Y):
    global model

    # Model training: logistic regression
    print("----------------------------------------")
    print("Begin model training...")
    # Time mark
    time_mark = time.time()
    # Fit model
    model = model.fit(dev_X, dev_Y)
    return time_mark


# Dev
def dev_main():
    global dev_features_vector_list_dataset, labels_list
    global model

    print("----------------------------------------")
    print("Begin developing...")
    print("----------------------------------------")

    # Read features and labels
    print("Reading features and labels...")
    dev_features_vector_list_dataset = np.load("./input/features/dev_features.npy", allow_pickle=True)
    labels_list = np.load("./input/labels/dev_labels.npy", allow_pickle=True)

    print("Length of dev dataset:", len(dev_features_vector_list_dataset))
    print("Length of dev labels list:", len(labels_list))

    # Get batch, which combines all frames together
    dev_X, dev_Y = generate_batch()
    print("Length of dev_X for model training:", len(dev_X))
    print("Length of dev_Y for model training:", len(dev_Y))

    # Model training, if you don't want to train this model again, just comment it!
    time_mark = model_train(dev_X, dev_Y)

    # Evaluate by using get_metrics() in evaluate.py, if you don't want to evaluate this model this time, just comment it!
    avg_auc, avg_eer = evaluate_dev()
    print("Model fitting completed, with solver in Logistic Regression is:", solver_name, "| Totally",
          str(round(time.time() - time_mark, 3)), "seconds spent...")
    print("Average AUC Score:", avg_auc, "| Average ERR Score:", avg_eer)

    # Save model, if you don't want to save this model this time, just comment it!
    print("----------------------------------------")
    print("Saving model...")
    joblib.dump(model, "./model_save/model.pkl")


# Test
def test_main():
    global test_features_vector_list_dataset
    # Predicted result
    predicted_result = []
    predicted_voice_curve = []

    print("----------------------------------------")
    print("Begin testing...")
    print("----------------------------------------")

    # Read model
    print("Reading model...")
    print("----------------------------------------")
    my_model = joblib.load("./model_save/model.pkl")

    # Read features
    print("Reading features and labels...")
    test_features_vector_list_dataset = np.load("./input/features/test_features.npy", allow_pickle=True)

    print("Length of test features vector list:", len(test_features_vector_list_dataset))

    # Time unit for each frame, need to be synchronized with time_unit in data_construction.py
    time_unit = 30
    # Time shift
    time_shift = int(time_unit / 2)

    # Predict
    print("----------------------------------------")
    print("Predicting labels on test dataset...")
    for i in range(len(test_features_vector_list_dataset)):
        if i % 100 == 0:
            print("Iteration:", str(i), "/", str(len(test_features_vector_list_dataset)), "rounds")
        my_predict = my_model.predict_proba(test_features_vector_list_dataset[i])
        predicted_result.append(my_predict)
        voice_curve = []
        for j in range(len(predicted_result[i])):
            # Curve[1] represents the voice! 1 for speaking, 0 for not speaking
            voice_curve.append(predicted_result[i][j][1])

        # Smooth the predicted voice curve: Median filter
        voice_curve = signal.medfilt(voice_curve, kernel_size=filter_size)
        # Normalized the predicted voice curve into 0 / 1 array
        for j in range(len(voice_curve)):
            voice_curve[j] = 1 if voice_curve[j] > voice_threshold else 0

        predicted_voice_curve.append(voice_curve)

    # Top five wav files in sorted test dataset: '104-132091-0020.wav', '104-132091-0028.wav', '104-132091-0041.wav',
    #                                               '104-132091-0050.wav', '104-132091-0061.wav'

    # Write
    write_path = "./output/test_prediction.txt"
    # write_path = "./output/train_prediction.txt"
    print("----------------------------------------")
    print("Writing result into path: \"" + write_path + "\" ...")
    final_result = []
    wav_files = []
    # Get the names of wav file
    wav_files = os.listdir("../vad/wavs/test")
    # wav_files = os.listdir("../vad/wavs/train")
    # Sort the names of wav file, note that when test the accuracy of prediction on test dataset!!!
    wav_files.sort()

    print("----------------------------------------")
    print("Begin random human test...")
    # Randomly choose a wav file to perform human test
    test_idx = secret_generator.randint(0, len(wav_files) - 1)
    print("Wav file", test_idx, ":", wav_files[test_idx], "(Please search this file in ../vad/wavs/test,")
    print(" and make artificial judge between this wav figure and predicted curve)")        # Print file name
    print("----------------------------------------")
    # plt.plot(predicted_result[0])
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


def train_main():
    global train_features_vector_list_dataset, labels_list
    predicted_result = []
    predicted_voice_curve = []

    print("----------------------------------------")
    print("Begin evaluating on train dataset (only takes 500 wav files)...")
    print("----------------------------------------")

    # Read model
    print("Reading model...")
    print("----------------------------------------")
    my_model = joblib.load("./model_save/model.pkl")

    # Read features
    print("Reading features and labels...")
    train_features_vector_list_dataset = np.load("./input/features/train_features.npy", allow_pickle=True)
    labels_list = np.load("./input/labels/train_labels.npy", allow_pickle=True)

    print("Length of train features vector list:", len(train_features_vector_list_dataset))

    # Time unit for each frame, need to be synchronized with time_unit in data_construction.py
    time_unit = 30
    # Time shift
    time_shift = int(time_unit / 2)

    # Predict
    print("Predicting labels on train dataset...")
    auc_score = 0
    eer_score = 0

    for i in range(len(train_features_vector_list_dataset)):
        if i % 100 == 0 and i != 0:
            print("Iteration:", str(i), "/", str(len(train_features_vector_list_dataset)),
                  "rounds | Current average AUC:", float(auc_score / i), "| Current average EER:", float(eer_score / i))
        voice_prob = my_model.predict_proba(train_features_vector_list_dataset[i])
        pred_Y = []
        for j in range(len(voice_prob)):
            pred_Y.append(voice_prob[j][1])

        # Smooth the predicted curve: Median filter
        pred_Y = signal.medfilt(pred_Y, kernel_size=filter_size)
        # Normalized the predicted voice curve into 0 / 1 array
        for j in range(len(pred_Y)):
            if pred_Y[j] == 0:
                # Silence State
                continue
            # Speech or Noise State
            pred_Y[j] = 1 if pred_Y[j] > voice_threshold else 0

        label_Y = labels_list[i][1]
        cur_auc, cur_eer = get_metrics(pred_Y, label_Y)
        auc_score += cur_auc
        eer_score += cur_eer

        predicted_voice_curve.append(pred_Y)

    print("Average AUC Score:", float(auc_score / len(train_features_vector_list_dataset)), "| Average EER Score:", float(eer_score / len(train_features_vector_list_dataset)))

    # Write
    write_path = "./output/train_prediction.txt"
    print("----------------------------------------")
    print("Writing result into path: \"" + write_path + "\" ...")
    final_result = []
    wav_files = []
    # Get the names of wav file
    wav_files = os.listdir("../vad/wavs/train")
    # Sort the names of wav file, note that when test the accuracy of prediction on test dataset!!!
    wav_files.sort()

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


def main():
    # Dev main, if you don't want to run developing method (model train and evaluate), just comment it!
    dev_main()

    # # Test main, if you don't want to run testing method (generate predictions on test dataset), just comment it!
    # test_main()

    # # Train main, only for local test for AUC and ERR (since it's not used in task 1),
    # # just comment it if you don't want to evaluate the model on train dataset!
    # train_main()


if __name__ == '__main__':
    main()
