import numpy as np
from random import choice
import matplotlib.pyplot as plt

# Load your data
data_small = np.load('content/data_small.npy')
label_small = np.load('content/label_small.npy')

data_large = np.load('content/data_large.npy')
label_large = np.load('content/label_large.npy')

data_small = np.hstack((data_small[:, 1:], np.ones((data_small.shape[0], 1))))
data_large = np.hstack((data_large[:, 1:], np.ones((data_large.shape[0], 1))))

MAX_ITERATION = 100
TOTAL_ITERATION = 0
def train_perceptron(training_data):
    global TOTAL_ITERATION
    # trainingdata: a list of datapoints, where training_data[0] contains
    # the data points and training_data[1] contains labels.
    # lanels are +1/-1 return learned model vector

    x = training_data[0]
    y = training_data[1]
    model_size = x.shape[1]
    # Get Random Weight
    w = np.zeros(model_size)
    iteration = 1

    canIterate = True
    #hypothesis set: np.sign(np.dot(x[i], w))
    while canIterate:
        missClassified = False

        for i in range(len(x)):
            # Check if its missclassified 
            # print("Hypothesis for iteration {}:{} ".format(i,np.sign(np.dot(x[i],w))))
            # np.dot(x[i],w) * y[i] <= 0 is same as line below
            if np.sign(np.dot(x[i],w)) != np.sign(y[i]):
                missClassified = True
                #Update weight
                w += y[i] * x[i]
                # print("missclassified")
                break
            if not missClassified and i is not len(x) - 1:
                continue
            if not missClassified and i == len(x) - 1:
                #if there is no missclasified break while loop
                canIterate = False
                print("Classification completed. After {} iteration".format(iteration))
                TOTAL_ITERATION += iteration
                break

        if iteration > MAX_ITERATION:  # Add convergence criteria to prevent infinite loop
            print("Perceptron did not converge after {} iterations".format(MAX_ITERATION))
            break
        iteration += 1
    return w

def print_prediction(model, data):
    
    result = np.matmul(data, model)
    predictions = np.sign(result)
    for i in range(len(data)):
        print("{}: {} -> {}".format(data[i][:2], result[i], predictions[1]))

def plot_data_and_boundary(x, y, w):
    plt.figure(figsize=(8, 6))

    # Plot data points.
    #if y is 1, take the point to the positive class if not to the negative class.
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='blue', label='Positive class')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], color='red', label='Negative class')

    # Plot linear separator => (w[0]*x + w[1]*y + w[2] = 0)
    x_values = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
    y_values = (-w[2] - w[0] * x_values) / w[1]
    plt.plot(x_values, y_values, color='green', label='Line Seperator')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptron')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    rnd_x = np.array(data_small)
    rnd_y = np.array(label_small)
    rnd_data = [rnd_x, rnd_y]

    test_amount = 1
    for _ in range(test_amount):
        trained_model = train_perceptron(rnd_data)
        print_prediction(trained_model, rnd_x)
        plot_data_and_boundary(rnd_x[:, :2], rnd_y, trained_model)

