import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.youtube.com/watch?v = pDdP0TFzsoQ & ab_channel = PythonEngineer
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py  # L35-L56
# https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html
# https://stackoverflow.com/questions/62319228/number-of-instances-per-class-in-pytorch-dataset
# https://deeplizard.com/learn/video/bH9Nkg7G8S0
# Stack Overflow...

# ============================================================================
class CNN(nn.Module):
	def __init__(self, num_classes): # init object
		super().__init__() # inherent nn.Module methods
		self.block1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.block2 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.fc_layers = nn.Sequential(
			nn.Linear(32*5*5, 64),
			nn.ReLU(),
			nn.Dropout(p = 0.2),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Dropout(p = 0.2),
			nn.Linear(32, num_classes))

	def forward(self, x):
		# Convolution Layers
		x = self.block1(x)
		x = self.block2(x)

		# Fully Connected Layers after flattening
		x = x.view(x.size(0), -1)
		x = self.fc_layers(x)

		return x
# ============================================================================
def main():

	# ============= Data Exploration ==============
	bs = 25
	# defining the transform as 1) converting it from X to tensor and 2) normalizing it to [-1,-1]. This is good because the range of values vary significally
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	# Importing CIFAR10 and creating batches
	trainset = torchvision.datasets.CIFAR10("/projects/brfi3983/", train=True, download=True, transform=transform)
	testset = torchvision.datasets.CIFAR10("/projects/brfi3983/", train=False, download=True, transform=transform)
	train = DataLoader(dataset=trainset, batch_size=bs,shuffle=True, num_workers=2)
	test = DataLoader(dataset=testset, batch_size=bs, shuffle=False, num_workers=2)

	# Exploring input data shape
	print(f'\nTraining Set Shape:{trainset.data.shape}')
	print(f'Test Set Shape:{testset.data.shape}')
	print(np.array(trainset.targets).shape)

	# Classes exploration
	print(f'Length of Classes: {len(trainset.classes)}')
	print(f'Classes: {trainset.classes}')
	instances = dict(Counter(trainset.targets))
	print(f'Instances of each class: {instances}')
	print(trainset.class_to_idx)

	# Plotting Instances
	plt.bar(instances.keys(), instances.values())
	plt.xlabel('Class')
	plt.ylabel('Count')
	plt.title('Class Instances')

	# Sample Image and label
	random_int = np.random.randint(0, 50000)
	image, label = trainset[random_int]
	image = image / 2 + 0.5 # This is to undo our normalizing
	print(f'Input Image Shape: {image.shape}\n')
	
	# Showing Sample Image
	image_t = image.permute(1, 2, 0)
	print(f'This is a {trainset.classes[label]}!')
	plt.figure()
	plt.imshow(image_t)
	plt.show()
	exit()
	# ============= Model Exploration ==============

	# Search for GPU (to use)
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		print("GPU activated.")
	else:
		device = torch.device("cpu")
		print("CPU activated.")

	# Hyperparameters/constants
	num_classes = 10
	learning_rate = 0.001
	epochs = 20
	steps = len(train)
	losses = []

	# Setting up the Model
	model = CNN(num_classes)
	model = model.to(device)
	print('=> Model Created')
	print(model) # Model Architecture
	criterion = nn.CrossEntropyLoss() # Already takes into account logits and softmax
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	# Training
	model.train()
	print('=> Training Started')
	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(train):
			# Grabbing batch
			X_train, y_train = data

			# Sending data to gpu device for training
			X_train, y_train = X_train.to(device), y_train.to(device)

			# Passing Data through Model and Calculating Loss
			y_pred = model(X_train)
			loss = criterion(y_pred, y_train)

			# Backward pass and updating weights
			optimizer.zero_grad() # need to zero so weights do not accumulate
			loss.backward() # total gradients
			optimizer.step()
			running_loss += loss.item()*X_train.size(0) # normalize batch loss to epoch loss

			# Printing stats
			if (i + 1) % 100 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, steps, loss.item()))
		
		# Storing normalized losses in list
		losses.append(running_loss / steps)
	print('=> Training Finished')

	# Grabbing a weight for a layer
	relu1_w = model.block1[0].weight.cpu().detach()
	print(f'Sample weight access after training:\n{relu1_w.numpy()[0,0,:,:]}')

	# Testing
	model.eval() #used to freeze batch norm and dropout (does not do anything for gradients!)
	print('=> Testing Started')

	# Using torch.no_grad() to not store tensor operations
	with torch.no_grad():
		correct = 0
		total = 0
		for i, data in enumerate(test):

			# Mini batch again
			X_test, y_test = data

			# Sending data to gpu device for testing
			X_test, y_test = X_test.to(device), y_test.to(device)

			# Passing though data and calculating accuracy
			y_pred = model(X_test)
			_, predicted = torch.max(y_pred.data, 1)
			total += y_test.size(0)
			correct += (predicted == y_test).sum().item()

		# Stats
		print('=> Testing Ended')
		print('Test Accuracy of the model on the 10000 test images: {} %'.format(
			100 * correct / total))

	# Printing training curve loss (will expand to validation soon)
	plt.figure()
	plt.plot(np.arange(0, epochs), losses)
	plt.xlabel('Epochs')
	plt.ylabel('Training Loss')
	plt.title('Model Loss')
	plt.show()

# ============================================================================
if __name__ == "__main__":
	main()
