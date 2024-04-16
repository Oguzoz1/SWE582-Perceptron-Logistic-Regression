import numpy as np
from random import choice
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff

############################# IMPORT AND SET DATA ################################
# Enter your path here:
arff_file = arff.loadarff('content/Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(arff_file[0])
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Target Class labels to 0 and 1. if Cammeo then 1 if not 0
y = np.where(y == b'Cammeo', 1, 0)
# Update the regarded column
df.iloc[:, -1] = y

################################ DATA SPLIT ######################################
# Shuffling the dataset according to a seed.
seed = 9
np.random.seed(seed)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_shuffle = X[indices]
y_shuffle = y[indices]
#Update the set
df.iloc[:, :-1] = X_shuffle
df.iloc[:, -1] = y_shuffle
print(df.head())

# Setting up train and test sets.
train_size = int(0.8 * X_shuffle.shape[0])
X_train = X_shuffle[:train_size]
y_train = y_shuffle[:train_size]
X_test = X_shuffle[train_size:]
y_test = y_shuffle[train_size:]

# Normalization
mean = np.mean(X_train)
std = np.std(X_train)
X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std
X_shuffle_normalized = (X_shuffle - mean) / std


########################### LOGISTIC REGRESSION #####################################

class LogisticRegression():
    def __init__(self, regularization = 0,learningRate = 0.001, iteration = 1000):
        self.learningRate = learningRate
        self.iteration = iteration
        self.regularization = regularization
        self.weights = None
        self.bias = None
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        costs = []
        for _ in range(self.iteration):
            pred = self.pred(X)

            # Computing regularized loss
            regularization = self.regularization / (2 * n_samples) * np.sum(self.weights**2)
            cost = self.logistic_loss(self.weights, X, y) + regularization
            costs.append(cost)

            # Gradient Descent Computation with regularization added  (Check source A1)
            uWeight = (1/n_samples) * np.dot((pred - y), X).T / len(X) + (self.regularization / n_samples) * self.weights
            uBias = (1/n_samples) * np.sum(pred - y) / len(X)

            # Update weight: moving opposite direction: v = -g (Check source A1)
            self.weights = self.weights - self.learningRate * uWeight
            self.bias = self.bias - self.learningRate * uBias

        return self.weights

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def logistic_loss(self, w, X, y):
        z = np.dot(X, w)
        loss = np.mean(np.log(1 + np.exp(-y * z)))
        return loss
    
    def accuracy(self,X,y):
        pred = self.pred(X)
        pred = pred > 0.5
        pred = np.array(pred)
        a = (1 - np.sum(np.absolute(pred - y)) / y.shape[0])*100
        print("Accuracy of the model is: " , a, "%")
        return a

    def pred(self, X):
        model = np.dot(X,self.weights) + self.bias
        pred = self.sigmoid(model)
        return pred
    
    def fold(self,X,y,regularization):
        fold_num = 5
        fold_accuracy = []
        fold_size = len(X) // fold_num

        for i in range(fold_num):
            x1 = i * fold_size
            x2 = (i + 1) * fold_size

            #Splitting into folds
            X_test_fold = X[x1: x2]
            y_test_fold = y[x1: x2]

            X_train_fold = np.concatenate((X[ : x1], X[x2 : ]))
            y_train_fold = np.concatenate((y[ : x1], y[x2 : ]))

            for reg_value in regularization:
                reg = LogisticRegression(reg_value, 0.001,1000)
                reg.train(X_train_fold,y_train_fold)
                print("Validation Step: ", i, "for regularization value of: ", reg_value)
                accuracy = reg.accuracy(X_test_fold, y_test_fold)
                fold_accuracy.append(accuracy)

        return fold_accuracy





#-155, -170, -250

# Model Test
logisticReg = LogisticRegression(1000,0.01,1000)
logisticReg.train(X_train_normalized,y_train)
test_accuracy = logisticReg.accuracy(X_test_normalized, y_test)

fold_accuracies = logisticReg.fold(X_shuffle_normalized, y_shuffle, [100, -110, 120, -90, 80])

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Regularization': [100, -110, 120, -90, 80],
    'Training Accuracy': [logisticReg.accuracy(X_train_normalized, y_train)] * 5, 
    'Test Accuracy': [test_accuracy] * 5,
})

# Display the results DataFrame
print(results_df)