import secrets
from Project1.Utils import *
import warnings
import sklearn
import joblib

# Close warnings
warnings.filterwarnings("ignore")

# By default
frame_size = 512  # Width of window
frame_shift = int(frame_size / 2)  # Shift of window
sample_rate = 16000     # Sample rate


# Reconstructed dev dataset of features vector list
dev_features_vector_list_dataset = []
# Reformatted dev labels vector
labels_list = []

# Secret generator
secret_generator = secrets.SystemRandom()

# Logistic Regression Model
model = sklearn.linear_model.LogisticRegression()


# Dev
def dev_main():
    global dev_features_vector_list_dataset, labels_list

    # Read features and labels
    dev_features_vector_list_dataset = np.load("./input/features/dev_features.npy", allow_pickle=True)
    labels_list = np.load("./input/labels/dev_labels.npy", allow_pickle=True)

    # Model training: logistic regression
    print("Begin model training...")
    for i in range(len(dev_features_vector_list_dataset)):
        if i % 10 == 0:
            print("Iteration", str(i), "/", str(len(dev_features_vector_list_dataset)), "| Training accuracy score:", str(model.score(dev_features_vector_list_dataset[i], labels_list[i][1])))
        model.fit(dev_features_vector_list_dataset[i], labels_list[i][1])

    # Save model
    print("Saving model...")
    joblib.dump(model, "./model_save/model.pkl")

    # Read model
    # print("Reading model...")
    # my_model = joblib.load("./model_save/model.pkl")
    # print(my_model.score(dev_features_vector_list_dataset[1], labels_list[1][1]))
    # print(my_model.predict(dev_features_vector_list_dataset[1]))


def main():
    # Dev
    # dev_main()
    global dev_features_vector_list_dataset, labels_list

    # Read features and labels
    dev_features_vector_list_dataset = np.load("./input/features/dev_features.npy", allow_pickle=True)
    labels_list = np.load("./input/labels/dev_labels.npy", allow_pickle=True)
    print("Length of dev dataset:", len(dev_features_vector_list_dataset))
    print("Length of dev labels list:", len(labels_list))


if __name__ == '__main__':
    main()
