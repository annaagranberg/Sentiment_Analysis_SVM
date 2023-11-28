import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

loaded_data = np.load('encoded_data.npz')

train_x_0 = loaded_data['train_x_0']
train_y_0 = loaded_data['train_y_0']
test_x_0 = loaded_data['test_x_0']
test_y_0 = loaded_data['test_y_0']

train_x_1 = loaded_data['train_x_1']
train_y_1 = loaded_data['train_y_1']
test_x_1 = loaded_data['test_x_1']
test_y_1 = loaded_data['test_y_1']

# X är (4588, 300) där 4588 är antalet dokument och 300 är längden på vektorn som representerar ett dokument 
# Y är (4588, 1) och representerar integers (0, 1, 2) alltså typen av bias.

# print(train_y_0[0:10]) -> [1,1,1,2,1,0,0,2,0,0]

class SVM:
    
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=10): # Får samma resultat med 1000 & 100
       
        self.lr = learning_rate  # Learning rate for gradient descent
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters  # Number of iterations for training
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):

        print("y shape: ", y.shape)
        
        n_samples, n_features = X.shape  # Get the number of samples and features in the training data
        self.w = np.zeros((len(np.unique(y)), n_features))  # Initialize the weight matrix
        self.b = np.zeros(len(np.unique(y)))  # Initialize the bias vector

        for class_label in np.unique(y):
         # Set class labels to -1 for samples not belonging to the current class
            binary_labels = np.where(y == class_label, 1, -1)
            for _ in range(self.n_iters):  # Total iterations
                for idx, x_i in enumerate(X):  # Each sample
                    # Compute the condition based on the hinge loss
                    condition = binary_labels[idx] * (np.dot(x_i, self.w[class_label].T) - self.b[class_label]) >= 1
                    if condition.all():
                        # If the condition holds for all samples, update weights with regularization term
                        self.w[class_label] = self.w[class_label] - self.lr * (2 * self.lambda_param * self.w[class_label])
                    else:
                        # If the condition is not met, update weights and bias to account for misclassifications
                        self.w[class_label] = self.w[class_label] - self.lr * (2 * self.lambda_param * self.w[class_label] - np.outer(binary_labels[idx], x_i)) 
                        self.b[class_label] = self.b[class_label] - self.lr * binary_labels[idx]


    def predict(self, X):
    # Return the index of the class with the highest score for each sample
        return np.argmax(np.dot(X, self.w.T) - self.b, axis=1)

# Initialize and train the SVM
svm_model = SVM()
svm_model.fit(train_x_0, train_y_0)

# Make predictions
predictions = svm_model.predict(test_x_0)

# Evaluate the model
accuracy = np.mean(predictions == test_y_0)  # Compute accuracy by comparing predictions with true labels
print(f"Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy of the trained SVM model

# PCA
pca = PCA(n_components=3)  
test_x_0_pca = pca.fit_transform(train_x_0)
train_x_0_pca = pca.transform(test_x_0)

svm_model = SVM()
svm_model.fit(train_x_0_pca, train_y_0)

predictions = svm_model.predict(test_x_0_pca)

accuracy = np.mean(predictions == test_y_0)
print(f"Accuracy with PCA: {accuracy * 100:.2f}%")