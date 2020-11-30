import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
from torchvision import models
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
# ============================================================================
plt.style.use('ggplot')
pre_model = True
bs = 20
num_classes = 21
lr = 1e-3

# Fully Convolutional Neural Network
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
		self.classifier = nn.Sequential(
			nn.Conv2d(num_classes, kernel_size=1),
			nn.ConvTranspose2d())

	# Forward Pass
	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.classifier(x)

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
		self.transform_VOC = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	# Downloading the data
	def prepare_data(self):
		self.full_dataset = torchvision.datasets.VOCSegmentation("/projects/brfi3983/", image_set='train', download=True, transform=transform_VOC)
		self.test_dataset = torchvision.datasets.VOCSegmentation("/projects/brfi3983/", image_set='val', download=True, transform=transform_VOC)

	# Creating train batches
	def train_dataloader(self):
		return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=True)

	# Creating validation batches
	def val_dataloader(self):
		return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)

	# Creating test batches
	def test_dataloader(self):
		return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)
# ============================================================================
def main():

	if pre_model == False:

		# Data Module and Model
		dm = DataModule(bs)
		dm.prepare_data()

		model = FCN(num_classes, learning_rate)

		# Running our Model
		trainer = Trainer(max_epochs=epochs, fast_dev_run=False, gpus=1, profiler=False, progress_bar_refresh_rate=1, logger=tboard_logger)
		trainer.fit(model, dm)


	if pre_model == True:

		model = models.segmentation.fcn_resnet101(pretrained=True).eval()

		transform_VOC = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		full_dataset = torchvision.datasets.VOCSegmentation("/projects/brfi3983/", image_set='train', download=True, transform=transform_VOC)
		test_dataset = torchvision.datasets.VOCSegmentation("/projects/brfi3983/", image_set='val', download=True, transform=transform_VOC)

		# # Loading our custom image and transforming it to our pretrained network (uncomment which image to use)
		# image = Image.open('bird.jpg')
		# image = Image.open('person.jpg')
		image = Image.open('dog.jpg')

		trf = T.Compose([T.Resize(800),
						T.CenterCrop(720),
						T.ToTensor(),
						T.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])])

		# Convert single image to single batch
		inp = trf(image).unsqueeze(0)
		out = model(inp)['out']
		print(f'Output shape: {out.shape}')

		# Seeing which classes are most dominant across its depth
		om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

		# Plotting Class distribution
		classes = np.unique(om)
		plt.hist(om, bins=num_classes)
		plt.title('Occurrences of Classes')
		plt.xlabel('Class')
		plt.ylabel('Count')

		# Plotting original image across segmented image
		fig, (ax1, ax2) = plt.subplots(1, 2)
		plt.suptitle('Image Segmentation')
		ax1.set_title('Original Image')
		ax1.imshow(image)
		ax2.set_title('Segmented Image')
		ax2.imshow(om, cmap='YlOrBr')

		plt.show()

# ============================================================================
if __name__ == "__main__":
	main()

