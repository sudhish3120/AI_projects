# version 1.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork
from operations import *
from sklearn.model_selection import KFold

def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t

def trainExperiment():
    X, y = load_dataset("data/wine_quality.csv", "quality")

    n_features = X.shape[1]
    net = NeuralNetwork(n_features, [32,32,16,1], [ReLU(), ReLU(), Sigmoid(), Identity()], MeanSquaredError(), learning_rate=0.001)
    epochs = 500

    test_split = 0.1
    X_train = X[:int((1 - test_split) * X.shape[0])]
    X_test = X[int((1 - test_split) * X.shape[0]):]
    y_train = y[:int((1 - test_split) * y.shape[0])]
    y_test = y[int((1 - test_split) * y.shape[0]):]

    trained_W, epoch_losses = net.train(X_train, y_train, epochs)
    print("Error on test set: {}".format(net.evaluate(X_test, y_test, mean_absolute_error)))

    plt.plot(np.arange(0, epochs), epoch_losses)
    plt.show()

def executeCrossValidation():
    X, y = load_dataset("data/wine_quality.csv", "quality")

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=1) # Ensure reproducibility with a fixed random state
    epochs = 500
    learning_rate = 0.001

    layer_sizes = [32, 32, 16, 1]
    activations = [ReLU(), ReLU(), Sigmoid(), Identity()]
    loss_function = MeanSquaredError()

    fold_maes = []
    folds_epoch_losses = np.zeros((epochs,))
    for train_i, val_i in kf.split(X):
        X_train, X_test = X[train_i], X[val_i]
        y_train, y_test = y[train_i], y[val_i]

        net = NeuralNetwork(X_train.shape[1], layer_sizes, activations, loss_function, learning_rate)

        trained_W, epoch_losses = net.train(X_train, y_train, epochs)

        mae = net.evaluate(X_test, y_test, mean_absolute_error)
        fold_maes.append(mae)

        folds_epoch_losses += np.array(epoch_losses)

    average_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    print(f"Average MAE: {average_mae}, Standard Deviation of MAE: {std_mae}")

    average_epoch_losses = folds_epoch_losses / k

    plt.plot(np.arange(1, epochs + 1), average_epoch_losses, label='Average Training Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Average Training Loss')
    plt.title('Average Training Loss Over Epochs Across All Folds')
    plt.legend()
    plt.show()
    


trainExperiment()
executeCrossValidation()