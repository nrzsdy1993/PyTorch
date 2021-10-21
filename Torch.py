#torch GPU 설정

import torch

if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')

#MNIST 데이터 셋
from torchvision import datasets

PATH_DATA = "./data"
train_data = datasets.MNIST(PATH_DATA, train = True, download=True)

#extract the input data and target labels
X_train, y_train = train_data.data, train_data.targets
print(X_train.shape)
print(y_train.shape)

#Load the MNIST test dataset
val_data = datasets.MNIST(PATH_DATA, train = False, download=True)
X_val, y_val = val_data.data, val_data.targets
print(X_val.shape)
print(y_val.shape)

#Add a new dimension to the tensors
if len(X_train.shape) == 3:
  X_train = X_train.unsqueeze(1)
print(X_train.shape)

if len(X_val.shape) == 3:
  X_val = X_val.unsqueeze(1)
print(X_val.shape)


#Visualization

from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def show(img):
  # convert tensor to numpy array
  np_img = img.numpy()

  # Convert to H * W * C shape
  np_img_tr = np.transpose(np_img, (1, 2, 0))

  plt.imshow(np_img_tr, interpolation='nearest')

X_grid = utils.make_grid(X_train[:20], nrow=4, padding=2)
print(X_grid.shape)

show(X_grid)
plt.show()