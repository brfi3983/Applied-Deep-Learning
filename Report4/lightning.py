import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
# ============================================================================

# Hyperparameters/constants
bs = 50
num_classes = 10
learning_rate = 0.001
epochs = 2

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
		accuracy = self.accuracy(outputs, y)

		# Logging to tensorboard (for each epoch)
		self.log('Training Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('Training Accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

		return {'loss': loss}

	# Runs during validation
	def validation_step(self, batch, batch_idx):
		x, y = batch
		outputs = self(x)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, y)
		accuracy = self.accuracy(outputs, y)

		# Logging to tensorboard (for each epoch)
		self.log('Validation Loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
		self.log('Validation Accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

		return {'loss': loss}

	# Runs during testing
	def test_step(self, batch, batch_idx):
		x, y = batch
		outputs = self(x)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(outputs, y)

	def training_epoch_end(self,outputs):

		# Logs computational Graph
		if self.current_epoch == 1:
			sample_img = torch.rand((1, 3, 32, 32), device=self.device)
			self.logger.experiment.add_graph(self, sample_img)

		# Logs weights and biases per each layer
		for name,params in self.named_parameters():
			self.logger.experiment.add_histogram(name, params, self.current_epoch)


class CIFAR10DataModule(pl.LightningDataModule):
	def __init__(self, bs):
		super().__init__()

		# Defined constants and transforms
		self.batch_size = bs
		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	# Downloading the data
	def prepare_data(self):
		self.full_dataset = torchvision.datasets.CIFAR10("/projects/brfi3983/", train=True, download=True, transform=self.transform)
		self.test_dataset = torchvision.datasets.CIFAR10("/projects/brfi3983/", train=False, download=True, transform=self.transform)

	# Splitting our data for our model
	def setup(self, stage=None):
		self.dims = (1, 3, 32, 32)
		self.train_dataset, self.val_dataset = random_split(self.full_dataset, [45000,5000]) # 50,000 "training" images = 45,000 training and 5,000 validaation

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

	# Logging
	tboard_logger = TensorBoardLogger('runs/cifar10', name='test run')
	wandb_logger = WandbLogger(name='test run', project='pytorch_lightning_test')

	# Data Module and Model
	dm = CIFAR10DataModule(bs)
	model = CNN(num_classes, learning_rate)

	# Running our Model
	trainer = Trainer(max_epochs=epochs, fast_dev_run=False, gpus=1, profiler=False, progress_bar_refresh_rate=1, logger=tboard_logger)
	trainer.fit(model, dm)

# ============================================================================
if __name__ == "__main__":
	main()

