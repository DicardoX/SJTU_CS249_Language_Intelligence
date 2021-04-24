import secrets
from Project1.Utils import *
import warnings
import sklearn
import joblib
from matplotlib import pyplot as plt
import secrets
import time

# Close warnings
# warnings.filterwarnings("ignore")

# Reconstructed dev dataset of features vector list
dev_features_vector_list_dataset = []
# Reconstructed test dataset of features vector list
test_features_vector_list_dataset = []
# Reformatted dev labels vector
labels_list = []

# Secret generator
secret_generator = secrets.SystemRandom()

# Solver for Logistic Regression
solver_name = "liblinear"

# Logistic Regression Model
# Solver="liblinear" is a good choice for small dataset
model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.001, random_state=123, solver=solver_name)

# Batch size
# batch_size = 1024


# Randomly choose a frame
def random_frame():
    # Randomly choose an index for wav
    idx_1 = secret_generator.randint(0, len(dev_features_vector_list_dataset) - 1)
    # Randomly choose an index for frame
    idx_2 = secret_generator.randint(0, len(dev_features_vector_list_dataset[idx_1]) - 1)
    return idx_1, idx_2


# Generate batch, put all frames together into a batch
def generate_batch():
    global dev_features_vector_list_dataset, labels_list  # , batch_size
    batch = []
    label = []

    for i in range(len(dev_features_vector_list_dataset)):
        for j in range(len(dev_features_vector_list_dataset[i])):
            batch.append(dev_features_vector_list_dataset[i][j])
            label.append(labels_list[i][1][j])

    # for i in range(batch_size):
    #     # Randomly choose a frame
    #     idx_1, idx_2 = random_frame()
    #     # Add into batch and label list
    #     batch.append(dev_features_vector_list_dataset[idx_1][idx_2])
    #     label.append(labels_list[idx_1][1][idx_2])
    return batch, label


# Dev
def dev_main():
    global dev_features_vector_list_dataset, labels_list

    # Read features and labels
    print("Reading features and labels...")
    dev_features_vector_list_dataset = np.load("./input/features/dev_features.npy", allow_pickle=True)
    labels_list = np.load("./input/labels/dev_labels.npy", allow_pickle=True)

    print("Length of dev dataset:", len(dev_features_vector_list_dataset))
    print("Length of dev labels list:", len(labels_list))

    # Model training: logistic regression
    print("----------------------------------------")
    print("Begin model training...")

    # Get batch, which combines all frames together
    dev_X, dev_Y = generate_batch()
    print("Length of dev_X:", len(dev_X))
    print("Length of dev_Y:", len(dev_Y))

    # Time mark
    time_mark = time.time()
    # Fit model
    classifier = model.fit(dev_X, dev_Y)

    # Accuracy
    idx_1, idx_2 = random_frame()
    idx_1 = 0
    acc_score = classifier.score(dev_features_vector_list_dataset[idx_1], labels_list[idx_1][1]) * 100
    print("Model fitting completed, with solver in Logistic Regression is:", solver_name, "| Totally", str(round(time.time() - time_mark, 3)), "seconds spent...")
    print("Accuracy (randomly choose a wav in dev dataset to test): %f%%" % acc_score)

    # count = 0
    # total_times = 5000
    # while count < total_times:
    #     count += 1
    #     # Randomly generate a batch
    #     dev_X, dev_Y = generate_batch()
    #     # Fit model
    #     model.fit(dev_X, dev_Y)
    #     # Info
    #     if count % 100 == 0:
    #         # Randomly choose a frame to test accuracy
    #         idx_1, idx_2 = random_frame()
    #         acc_score = model.score(dev_features_vector_list_dataset[idx_1], labels_list[idx_1][1])
    #         print("Iteration", str(count), "/", str(total_times), "| Training accuracy score:",
    #               str(acc_score))
    #         print(model.coef_)
    #         # Push into list
    #         accuracy_list.append(acc_score)

    # Save model
    print("Saving model...")
    joblib.dump(model, "./model_save/model.pkl")


# Test
def test_main():
    global test_features_vector_list_dataset
    # Predicted result
    predicted_result = []

    # Read model
    print("Reading model...")
    my_model = joblib.load("./model_save/model.pkl")

    # Read features
    print("Reading features and labels...")
    test_features_vector_list_dataset = np.load("./input/features/test_features.npy", allow_pickle=True)

    # Predict
    for i in range(10):
        predicted_result.append(my_model.predict_proba(test_features_vector_list_dataset[i]))

    curve_1 = []
    curve_2 = []
    test_idx = 4
    for i in range(len(predicted_result[test_idx])):
        curve_1.append(predicted_result[test_idx][i][0])
        curve_2.append(predicted_result[test_idx][i][1])

    # plt.plot(predicted_result[0])
    plt.plot(curve_2)
    plt.savefig("./output/predict")
    plt.show()

    # Top five wav files in sorted test dataset: '104-132091-0020.wav', '104-132091-0028.wav', '104-132091-0041.wav',
    #                                               '104-132091-0050.wav', '104-132091-0061.wav'

    # print(model.predict_proba(test_X))


def main():
    # # Dev main
    # dev_main()
    # Test main
    test_main()


if __name__ == '__main__':
    main()
