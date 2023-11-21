import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loaded_data = np.load('encoded_data.npz')

train_x_0 = loaded_data['train_x_0']
train_y_0 = loaded_data['train_y_0']
test_x_0 = loaded_data['test_x_0']
test_y_0 = loaded_data['test_y_0']

train_x_1 = loaded_data['train_x_1']
train_y_1 = loaded_data['train_y_1']
test_x_1 = loaded_data['test_x_1']
test_y_1 = loaded_data['test_y_1']

# Check the dimensions of the data
print(len(train_x_0), len(train_y_0))
print(train_x_0.shape, train_y_0.shape)

# Slutsatser: 

# X är (4588, 300) där 4588 är antalet dokument och 300 är längden på vektorn som representerar ett dokument 
# Y är (4588, 1) och representerar integers (0, 1, 2) alltså typen av bias.

# print(train_y_0[0:10]) -> [1,1,1,2,1,0,0,2,0,0]


#----------------- Plotting with PCA -----------------#

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

x_data = train_x_0
y_data = train_y_0

# Apply PCA to reduce x_data to 3 dimensions
pca = PCA(n_components=3)
x_3d = pca.fit_transform(x_data)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_3d[:, 0], x_3d[:, 1], x_3d[:, 2], c=y_data, cmap='viridis')

# Add labels and legend
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D Scatter Plot with PCA')

# Add colorbar for the class labels
cbar = plt.colorbar(scatter)
cbar.set_label('Class Label')

# Show the plot
plt.show()

#------------------------------------------------------#

#----------------- Support vector machine -----------------#