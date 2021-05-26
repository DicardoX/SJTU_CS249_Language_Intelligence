# train_GMM.py

import warnings
import sklearn
import joblib
from matplotlib import pyplot as plt
import secrets
import time
import random
from scipy import signal
import os
from evaluate import get_metrics
from sklearn import metrics
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from Project1_2.Utils import *

# Reconstructed dev dataset of features vector list
dev_features_vector_list_dataset = []
# Reconstructed test dataset of features vector list
test_features_vector_list_dataset = []
# Reconstructed train dataset of features vector list
train_features_vector_list_dataset = []
# Reformatted dev labels vector
labels_list = []

# GMM models num
models_num = 2
# Filter size
filter_size = 17
# Scale factor
scale_factor = 10000
# Batch size
batch_size = 512

# Secret generator
secret_generator = secrets.SystemRandom()


# Gaussian Mixture Model(GMM) with n_components = 2 since VAD is a binary classification
# Warm start: (used when Mini Batch GMM) use the last fitting result to initialize current fitting
#             (used when fit several times in program), will ignore n_init
# Construct GMM models list
def construct_GMM_models():
    models_list = []
    for i in range(models_num):
        my_model = GaussianMixture(n_components=16, tol=1e-4, reg_covar=1e-6, max_iter=500, init_params="kmeans", random_state=123,
                                   warm_start=False, verbose=1, verbose_interval=1)
        models_list.append(my_model)
    return models_list


# Generate batch, put all frames together into a batch
def generate_batch(features_vector_list_dataset, labels_list_dataset):
    batch = []
    label = []
    for i in range(len(features_vector_list_dataset)):
        for j in range(len(features_vector_list_dataset[i])):
            batch.append(features_vector_list_dataset[i][j])
            label.append(labels_list_dataset[i][1][j])
    # Shuffle
    idx = [i for i in range(len(batch))]
    random.shuffle(idx)
    shuffle_batch = []
    shuffle_label = []
    for i in range(len(idx)):
        shuffle_batch.append(batch[idx[i]])
        shuffle_label.append(label[idx[i]])
    return batch, label


# Classify batches: 0 for negative samples, 1 for positive samples
def classify_batch(train_X, train_Y):
    train_X_list = [[] for i in range(models_num)]
    train_Y_list = [[] for i in range(models_num)]

    for i in range(len(train_X)):
        if train_Y[i] == 0:
            train_X_list[0].append(train_X[i])
            train_Y_list[0].append(train_Y[i])
        elif train_Y[i] == 1:
            train_X_list[1].append(train_X[i])
            train_Y_list[1].append(train_Y[i])
    return train_X_list, train_Y_list


# Normalization
def normalization(features_num, train_X):
    # Normalization
    train_means = [0 for i in range(features_num)]
    train_std_min = [100000 for j in range(features_num)]
    train_std_max = [-100000 for j in range(features_num)]
    for i in range(len(train_X)):
        # Mean
        train_means = list(np.add(train_means, train_X[i]))
        # Std
        for j in range(features_num):
            train_std_max[j] = train_X[i][j] if train_std_max[j] < train_X[i][j] else train_std_max[j]
            train_std_min[j] = train_X[i][j] if train_std_min[j] > train_X[i][j] else train_std_min[j]
    train_means = np.divide(train_means, float(len(train_X)))
    delta_train_std = [0 for i in range(features_num)]
    for i in range(features_num):
        delta_train_std[i] = train_std_max[i] - train_std_min[i]
    for i in range(len(train_X)):
        train_X[i] = np.subtract(train_X[i], train_means)
        train_X[i] = np.divide(train_X[i], delta_train_std)
        train_X[i] = np.multiply(train_X[i], scale_factor)

    return train_X, train_means, delta_train_std, train_std_min


# Evaluate
def evaluate(models_list, valid_features_vector_list, valid_labels_list, means, delta_std):
    # Model likelihood
    print("---------------------------------------------------------------")
    print("Begin GMM model evaluation based on dev dataset...")

    auc_score, eer_score = 0, 0

    for i in range(len(valid_features_vector_list)):
        if i % 100 == 0 and i != 0:
            print("Iteration:", str(i), "/", str(len(valid_features_vector_list)),
                  "rounds | Current average AUC:", float(auc_score / i), "| Current average EER:", float(eer_score / i))

        valid_X = valid_features_vector_list[i]
        valid_Y = valid_labels_list[i][1]

        # Normalization
        for j in range(len(valid_X)):
            valid_X[j] = np.subtract(valid_X[j], means)
            valid_X[j] = np.divide(valid_X[j], delta_std)
            valid_X[j] = np.multiply(valid_X[j], scale_factor)

        pos_likelihood = list(models_list[1].score_samples(valid_X))
        neg_likelihood = list(models_list[0].score_samples(valid_X))

        # Smooth the likelihood curve
        # pos_likelihood = signal.medfilt(pos_likelihood, kernel_size=filter_size)
        # neg_likelihood = signal.medfilt(neg_likelihood, kernel_size=filter_size)

        # Model predict
        pred = [0 for j in range(len(pos_likelihood))]
        for j in range(0, len(pos_likelihood), 1):
            pred[j] = 0 if neg_likelihood[j] > pos_likelihood[j] else 1

        # Smooth the pred curve: Median filter
        pred = signal.medfilt(pred, kernel_size=filter_size)

        cur_auc, cur_eer = get_metrics(pred, valid_labels_list[i][1])

        auc_score += cur_auc
        eer_score += cur_eer

    return float(auc_score / len(valid_features_vector_list)), float(eer_score / len(valid_features_vector_list))


# Unit train of GMM model
def model_train(model, train_X):
    # Model training: Gaussian Mixture Model
    # -------------- Big batch GMM ------------------
    model = model.fit(train_X)
    # ------------- Mini batch GMM ------------------
    # idx = 0
    # for batch_idx in range(0, len(train_X), batch_size):
    #     model = model.fit(train_X[batch_idx:(batch_idx + batch_size)])
    #     idx = batch_idx
    # model = model.fit(train_X[idx:len(train_X)])

    return model


# Train
def train_main():
    # Models list
    models_list = construct_GMM_models()

    time_mark = time.time()

    # Read features and labels
    print("---------------------------------------------------------------")
    print("Reading features and labels...")
    features_vector_list_dataset = np.load("./input/features/train_features.npy", allow_pickle=True)
    labels_list_dataset = np.load("./input/labels/train_labels.npy", allow_pickle=True)

    # # of features
    features_num = len(features_vector_list_dataset[0][0])

    print("Length of train dataset:", len(features_vector_list_dataset))
    print("Length of train labels list:", len(labels_list_dataset))
    print("Length of features vector:", features_num,
          "(15 PLP features, 13 MFCC features, 1 ZCR feature, 1 STE feature)")

    # Get batch, which combines all frames together
    train_X, train_Y = generate_batch(features_vector_list_dataset, labels_list_dataset)
    print("Length of train_X for model training:", len(train_X))
    print("Length of train_Y for model training:", len(train_Y))

    # Normalization
    train_X, means, delta_std, std_min = normalization(features_num, train_X)
    np.save("./model/model_distribution.npy", [means, delta_std])

    # Classify train samples
    print("---------------------------------------------------------------")
    print("Classifying train samples into negative batch and positive batch...")
    train_X_list, train_Y_list = classify_batch(train_X, train_Y)
    print("Length of negative samples: %d | Length of positive samples: %d" % (
    len(train_X_list[0]), len(train_X_list[1])))

    # Model training, if you don't want to train this model again, just comment it!
    print("---------------------------------------------------------------")
    print("Begin GMM model training...")
    for i in range(models_num):
        print("---------------------------------------------------------------")
        message = "Iteration 1 / 2: Train for negative samples..." if i == 0 else "Iteration 2 / 2: Train for positive samples..."
        print(message)
        models_list[i] = model_train(models_list[i], train_X_list[i])

    # Model self evaluation
    valid_features_vector_list = np.load("./input/features/train_features.npy", allow_pickle=True)
    valid_labels_list = np.load("./input/labels/train_labels.npy", allow_pickle=True)
    auc, eer = evaluate(models_list, valid_features_vector_list, valid_labels_list, means, delta_std)

    # Save model, if you don't want to save this model this time, just comment it!
    print("---------------------------------------------------------------")
    print("Saving model, means and delta_std...")
    joblib.dump(models_list[0], "./model/neg_model.pkl")
    joblib.dump(models_list[1], "./model/pos_model.pkl")
    np.save("./model/model_distribution.npy", [[means], [delta_std]])
    print("---------------------------------------------------------------")
    print("GMM train completed! | Total time: %.2f seconds | Train AUC: %.4f | Train EER: %.4f" % (
    (time.time() - time_mark), auc, eer))


# Valid
def valid_main():
    print("---------------------------------------------------------------")
    print("Begin model validating...")

    # Read model
    print("---------------------------------------------------------------")
    print("Loading model...")
    pos_GMM = joblib.load("./model/pos_model.pkl")
    neg_GMM = joblib.load("./model/neg_model.pkl")
    models_list = [neg_GMM, pos_GMM]
    res = np.load("./model/model_distribution.npy", allow_pickle=True)
    means, delta_std = res[0], res[1]

    # Read features
    print("---------------------------------------------------------------")
    print("Reading features and labels...")
    valid_X = np.load("./input/features/dev_features.npy", allow_pickle=True)
    valid_Y = np.load("./input/labels/dev_labels.npy", allow_pickle=True)

    valid_auc, valid_eer = evaluate(models_list, valid_X, valid_Y, means, delta_std)

    print("---------------------------------------------------------------")
    print("GMM train completed! | Valid AUC: %.4f | Valid EER: %.4f" % (valid_auc, valid_eer))


if __name__ == '__main__':
    # # Train
    # train_main()

    # Valid
    valid_main()
