import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
# from torchvision.models import models
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
plt.style.use('ggplot')
# ============================================================================

# Hyperparameters/constants
bs = 1
num_classes = 10
learning_rate = 0.001
epochs = 99
data_prep = 0
# Convolutional Neural Network
class CNN(pl.LightningModule):
	def __init__(self, num_classes, learning_rate):
		super().__init__()

		# Defining Accuracy and Model Architecture
		self.accuracy = pl.metrics.Accuracy()
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
			nn.Dropout(p=0.2),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(32, num_classes))

	# Forward Pass
	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = x.view(x.size(0), -1)
		x = self.fc_layers(x)

		return x

	# Optimizer - can run multiple optimizers
	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=learning_rate)

		return optimizer

	# Runs during training
	def training_step(self, batch, batch_idx):
		x, y = batch
		outputs = self(x)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, y)

		return {'loss': loss}

	# Runs during validation
	def validation_step(self, batch, batch_idx):
		x, y = batch
		outputs = self(x)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, y)

		return {'loss': loss}

	# Runs during testing
	def test_step(self, batch, batch_idx):
		x, y = batch
		outputs = self(x)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, y)

class Resnet50(pl.LightningModule):
	def __init__(self, num_classes):
		super().__init__()

		# Defining Accuracy and Model Architecture
		self.num_classes = num_classes
		self.feature_extractor = models.resnet50(pretrained=True)
		self.feature_extractor.eval()
		self.classifier = nn.Sequential(nn.Linear(1000, 512),
										nn.ReLU(),
										nn.Dropout(0.4),
										nn.Linear(512, num_classes))
	# Forward Pass
	def forward(self, x):
		representations = self.feature_extractor(x)
		x = self.classifier(representations)
		return x

	# Optimizer - can run multiple optimizers
	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=learning_rate)

		return optimizer

	# Runs during training
	def training_step(self, batch, batch_idx):
		x, y = batch
		outputs = self(x)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, y)

		return {'loss': loss}

	# Runs during validation
	def validation_step(self, batch, batch_idx):
		x, y = batch
		outputs = self(x)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, y)

		return {'loss': loss}

	# Runs during testing
	def test_step(self, batch, batch_idx):
		x, y = batch
		outputs = self(x)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, y)

class DataModule(pl.LightningDataModule):
	def __init__(self, bs, data_str):
		super().__init__()

		# Defined constants and transforms
		self.batch_size = bs
		self.data_str = data_str
		self.transform_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])])
		self.transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		self.transform_imagenet = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	# Downloading the data
	def prepare_data(self):
		if self.data_str == 'cifar10':
			self.full_dataset = torchvision.datasets.CIFAR10("/projects/brfi3983/", train=True, download=True, transform=self.transform_cifar10)
			self.test_dataset = torchvision.datasets.CIFAR10("/projects/brfi3983/", train=False, download=True, transform=self.transform_cifar10)
		if self.data_str == 'mnist':
			self.full_dataset = torchvision.datasets.MNIST("/projects/brfi3983/", train=True, download=False, transform=self.transform_mnist)
			self.test_dataset = torchvision.datasets.MNIST("/projects/brfi3983/", train=False, download=False, transform=self.transform_mnist)
		if self.data_str == 'imagenet':
			self.full_dataset = torchvision.datasets.ImageNet("/projects/brfi3983/", train=True, download=False, transform=self.transform_imagenet)
			self.test_dataset = torchvision.datasets.ImageNet("/projects/brfi3983/", train=False, download=False, transform=self.transform_imagenet)
	# Splitting our data for our model
	def setup(self, stage=None):
		if self.data_str == 'mnist':
			self.dims = (1, 28, 28)
			self.train_dataset, self.val_dataset = random_split(self.full_dataset, [55000,5000]) # 60,000 "training" images = 55,000 training and 5,000 validaation
		if self.data_str == 'cifar10':
			self.dims = (1, 3, 32, 32)
			self.train_dataset, self.val_dataset = random_split(self.full_dataset, [45000,5000]) # 50,000 "training" images = 45,000 training and 5,000 validaation
		if self.data_str == 'imagenet':
			self.dims = (1, 3, 224, 224)
			self.train_dataset, self.val_dataset = random_split(self.full_dataset, [45000,5000]) # 50,000 "training" images = 45,000 training and 5,000 validaation ?

	# Creating train batches
	def train_dataloader(self):
		return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=True)

	# Creating validation batches
	def val_dataloader(self):
		return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)

	# Creating test batches
	def test_dataloader(self):
		return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)

def extract_conv_output(array):
	def hook(model, input, output):
		array['conv'] = output.detach()
	return hook
def extract_fc_output(array):
	def hook(model, input, output):
		array['fc'] = output.detach()
	return hook
# ============================================================================
def main():

	if data_prep == True:
		n = 10000 # number of samples for dimensionality reduction
		# # =================================================================== MNIST ============================================================

		# # Loading MNIST
		# dm = DataModule(bs, 'mnist')
		# dm.prepare_data()
		# dm.setup()

		# # Plotting the first 5 images from the training set
		# for i in range(0, 5):
		# 	plt.subplot(1, 5, i + 1)
		# 	plt.axis('off')
		# 	plt.suptitle('MNIST Dataset Samples...')
		# 	image, label = dm.train_dataset[i]
		# 	plt.imshow(image[0,:,:])

		# # Storing data in NumPy arrays
		# X = dm.train_dataset.dataset.data.numpy()
		# y = dm.train_dataset.dataset.targets.numpy()

		# # Grabbing the first n samples
		# X = X[0:n,:]
		# X = X.reshape(X.shape[0], -1) #flattening to R^n vector
		# y1 = y[0:n]

		# # Calculating TSNE and PCA
		# print('Doing TSNE and PCA for MNIST...')
		# model_tsne = TSNE(n_components=2, random_state=0)
		# model_pca = PCA(n_components=2, random_state=0)
		# tsne = model_tsne.fit_transform(X)
		# pca = model_pca.fit_transform(X)

		# # Creating a plot of subplots for PCA and TSNE
		# plt.suptitle('MNIST')
		# plt.subplot(1, 2, 1)
		# plt.title('PCA')
		# plt.xlabel('Dimension 1')
		# plt.ylabel('Dimension 2')
		# sns.scatterplot(x=pca[:,0],y=pca[:,1], hue=y1, palette=sns.color_palette("hls", 10), legend='full', alpha=0.5)
		# plt.subplot(1, 2, 2)
		# plt.title('TSNE')
		# plt.xlabel('Dimension 1')
		# plt.ylabel('Dimension 2')
		# sns.scatterplot(x=tsne[:,0],y=tsne[:,1], hue=y1, palette=sns.color_palette("hls", 10), legend='full', alpha=0.5)

		# # =================================================================== Cifar10 ============================================================

		# # Loading CIFAR10
		# dm = DataModule(bs, 'cifar10')
		# dm.prepare_data()
		# dm.setup()

		# plt.figure()
		# for i in range(0, 5):
		# 	plt.subplot(1, 5, i + 1)
		# 	plt.axis('off')
		# 	plt.suptitle('Cifar10 Dataset Samples...')
		# 	image, label = dm.train_dataset[i]
		# 	image = image / 2 + 0.5
		# 	image = image.permute(1, 2, 0)
		# 	plt.imshow(image[:,:,:])

		# # Storing data in NumPy arrays
		# X = dm.train_dataset.dataset.data
		# y = dm.train_dataset.dataset.targets

		# # Grabbing the first n samples
		# X = X[0:n,:]
		# X = X.reshape(X.shape[0], -1) #flattening to R^n vector
		# y1 = y[0:n]

		# # Calculating TSNE and PCA
		# print('Doing TSNE and PCA for Cifar10...')
		# model_tsne = TSNE(n_components=2, random_state=0)
		# model_pca = PCA(n_components=2, random_state=0)
		# tsne = model_tsne.fit_transform(X)
		# pca = model_pca.fit_transform(X)

		# # Creating a plot of subplots for PCA and TSNE
		# plt.suptitle('Cifar10')
		# plt.subplot(1, 2, 1)
		# plt.title('PCA')
		# plt.xlabel('Dimension 1')
		# plt.ylabel('Dimension 2')
		# sns.scatterplot(x=pca[:,0],y=pca[:,1], hue=y1, palette=sns.color_palette("hls", 10), legend='full', alpha=0.5)
		# plt.subplot(1, 2, 2)
		# plt.title('TSNE')
		# plt.xlabel('Dimension 1')
		# plt.ylabel('Dimension 2')
		# sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y1, palette=sns.color_palette("hls", 10), legend='full', alpha=0.5)

		# plt.show()

		# =================================================================== ImageNet ============================================================

		dm = DataModule(bs, 'imagenet')
		dm.prepare_data()
		dm.setup()

		print(dm.full_dataset.dataset)
		exit()
		plt.figure()
		for i in range(0, 5):
			plt.subplot(1, 5, i + 1)
			plt.axis('off')
			plt.suptitle('ImageNet Dataset')
			image, label = dm.train_dataset[i]
			# image = image / 2 + 0.5
			image = image.permute(1, 2, 0)
			plt.imshow(image[:,:,:])

		# IMAGENET??

		plt.show()

	else:
		# Model
		# model = Resnet50(num_classes)

		model = CNN.load_from_checkpoint(checkpoint_path="C:/Users/user/Documents/School/2020-2021/APPM 5720/Report4/checkpoints/epoch=99.ckpt")
		# first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
		# first_conv_layer.extend(list(model.features))
		# model.features= nn.Sequential(*first_conv_layer)
		model.freeze()

		print(model)
		conv1 = {}
		conv2 = {}
		fc1 = {}
		fc2 = {}
		fc3 = {}
		# model.feature_extractor.fc.register_forward_hook(extract_conv_output(results))
		model.block1[0].register_forward_hook(extract_conv_output(conv1))
		model.block2[0].register_forward_hook(extract_conv_output(conv2))
		model.fc_layers[0].register_forward_hook(extract_fc_output(fc1))
		model.fc_layers[3].register_forward_hook(extract_fc_output(fc2))
		model.fc_layers[6].register_forward_hook(extract_fc_output(fc3))

		# Loading in CIFAR10 Data
		dm = DataModule(bs, 'cifar10')
		dm.prepare_data()
		dm.setup()

		# Taking an example image
		image, label = dm.train_dataset[0]
		image = image.unsqueeze(0)
		predictions = np.argmax(model(image).detach().numpy())

		# Grabbing our Hooked Layers
		conv1_layer = conv1['conv'].cpu().numpy()
		conv2_layer = conv2['conv'].cpu().numpy()
		fc1_layer = fc1['fc'].cpu().numpy()
		fc2_layer = fc2['fc'].cpu().numpy()
		fc3_layer = fc3['fc'].cpu().numpy()
		print(f'Predicted Label: {predictions} | True Label: {label}')

		# Graphing the Image
		image_t = image.squeeze(0)
		image_t = image_t / 2 + 0.5
		image_t = image_t.permute(1, 2, 0)
		plt.title('Original Image')
		plt.imshow(image_t)


		# Visualizing the Convolution filters for the first conv. layer
		conv1_weight = model.block1[0].weight.detach().numpy()
		plt.figure()
		plt.suptitle('Convolution Filters')
		for i in range(0, conv1_weight.shape[0]):
			plt.subplot(8, 8, i + 1)
			plt.axis('off')
			plt.imshow(conv1_weight[i,0,:,:], cmap='gray')

		# Graphing First Feature Maps Layer
		plt.figure()
		plt.suptitle('Feature Maps in First Layer')
		for i in range(0, conv1_layer.shape[1]):
			plt.subplot(8, 8, i + 1)
			plt.axis('off')
			plt.imshow(conv1_layer[0,i,:,:], cmap='gray')

		# Graphing Second Feature Maps Layer
		plt.figure()
		plt.suptitle('Feature Maps in Second Layer')
		for i in range(0, conv2_layer.shape[1]):
			plt.subplot(4, 8, i + 1)
			plt.axis('off')
			plt.imshow(conv2_layer[0,i,:,:], cmap='gray')

		# Graphing the Fully Connected Layers
		print(f'Final Layer\'s Activations: {fc3_layer}')
		plt.figure()
		plt.axis('on')
		plt.suptitle('Fully Connected Layers')
		plt.subplot(3, 1, 1)
		plt.title('First Layer')
		plt.imshow(fc1_layer)
		plt.subplot(3, 1, 2)
		plt.title('Second Layer')
		plt.imshow(fc2_layer)
		plt.subplot(3, 1, 3)
		plt.title('Third Layer')
		plt.xticks(np.arange(0, 9, step=1))
		plt.colorbar()
		plt.imshow(fc3_layer)

		plt.show()

		exit()












		# model = CNN.load_from_checkpoint(checkpoint_path="C:/Users/user/Documents/School/2020-2021/APPM 5720/Report4/checkpoints/epoch=99.ckpt")

		# image, label = dm.train_dataset[0]

		# image_t = image / 2 + 0.5
		# image_t = image_t.permute(1, 2, 0)
		# plt.imshow(image_t)
		# image = image[np.newaxis, :, :, :]
		# print(model)
		# for name, module in model.named_modules():
		# 	print(name)
		# model.block1[0].register_forward_hook(lambda model, input, output: print(input, output))

		# Visualizing the Convolution filters for the first conv. layer
		# conv1_weight = model.block1[0].weight.detach().numpy()
		# print(conv1_weight.shape)
		# for i in range(0, conv1_weight.shape[0]):
		# 	plt.subplot(8, 8, i + 1)
		# 	plt.axis('off')
		# 	plt.imshow(conv1_weight[i,0,:,:], cmap='gray')
		# plt.show()

		# Visualizing our feature maps from the above convolution layer

		# feature_map_arr = model(image)
		plt.figure()
		plt.imshow(output)
		plt.show()

		# Use Pretrained VGG-19!
		# Report:
		# 1) Visualize Mnist, Cifar10, ImageNet classes with TSNE, UMAP, PCA
		# 2) VIsualize a sample image of each dataset's feature maps (first layer and mid layer?)
		# 3) Visualize the same image's fully connected layer with PCA (n=2)
# ============================================================================
if __name__ == "__main__":
	main()

