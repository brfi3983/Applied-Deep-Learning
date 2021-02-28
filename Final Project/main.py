import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

plt.style.use('ggplot')
# ============================================================================

# Hyperparameters/constants
bs = 1
learning_rate = 0.01

def features(image, model, layers):

	# Storing features in dictionary due to variable sizes
	features = {}

	# Recursively go through each module and take the name and layer
	out = image
	for name, layer in model._modules.items():
		out = layer(out)

		if name in layers:
			features[layers[name]] = out
	return features

def content_loss(target, c_features):

	c_loss = 0
	for feature in c_features:
		c_loss += torch.mean((target[feature] - c_features[feature])** 2)

	return c_loss

def style_loss(target, s_features, style_weights=None):

	s_loss = 0
	for feature in s_features:

		# getting dimensions for normalizing
		n, d, h, w = target[feature].shape

		# Calculate gram matrix for each style layer (and its target style)
		gram = gram_matrix(s_features[feature])
		t_gram = gram_matrix(target[feature])

		# Calculate style loss per layer and weight it
		layer_style_loss = style_weights[feature] * torch.mean((t_gram - gram)** 2)

		# Sum to total loss after normalizing
		s_loss += layer_style_loss / (d * h * w)

	return s_loss

def gram_matrix(feature):

	# unpacking shape (n=1)
	# print(feature.shape)
	n, d, h, w = feature.size()

	# Flattening 3D tensor into 2D matrix
	feature = feature.view(d, h * w)

	# Create gram matrix
	gram = torch.mm(feature, feature.t())

	return gram

def extract_output(dict, name):

	def hook(model, input, output):

		dict[name] = output.detach().cpu()

	return hook

def color_l(target, style_photo):

	n, d, h, w = style_photo.shape
	trf = T.Compose([T.Resize(h, w),
					T.ToTensor()])

	target = convert_image_colors(target.clone().detach())
	target = T.ToPILImage()(target) #CANT CHANGE TARGET - HAVE TO CHANGE BACK!
	target = trf(target)

	# Calculate gram matrix for each style layer (and its target style)
	gram = gram_matrix(style_photo)
	t_gram = gram_matrix(target.unsqueeze(0)).to('cuda')

	# Calculate style loss per layer and weight it
	color_loss = torch.mean((t_gram - gram)**2)

	return color_loss

def convert_to_image(image):

	image_show = image.clone().cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
	image_show = image_show * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
	image_show = image_show.clip(0, 1)

	return image_show

def convert_image_colors(image):

	image_show = image.cpu().detach().squeeze(0)

	return image_show
# ============================================================================
def main():

	# Style Attributes
	content_layer = {'34': 'conv5_4'}
	# content_layer = {'2': 'conv1_2'}
	style_layers = {'0': 'conv1_1',
				  '5': 'conv2_1',
				  '10': 'conv3_1',
				  '19': 'conv4_1',
				  '28': 'conv5_1'}

	s_weights = {'conv1_1': 1.5,
				 'conv2_1': 0.80,
				 'conv3_1': 0.3,
				 'conv4_1': 0.4,
				 'conv5_1': 0.4}
	# s_weights = {'conv1_1': 0.1,
	# 			 'conv2_1': 0.3,
	# 			 'conv3_1': 2.0,
	# 			 'conv4_1': 7.0,
	# 			 'conv5_1': 10.0}

	# ====================== HARDCODE THESE COMBINATIONS!!!!! =======================

	# Overall Loss Style vs. Content Weights
	c_weight = 1e3
	s_weight = 1
	color_weight = 0

	# Clearing Cude Memory and setting device as GPU
	torch.cuda.empty_cache()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Importing Original and style image
	image = Image.open('Dolomites.jpg')
	style_image = Image.open('shortstache_style.jpg')
	print(np.array(image).shape)

	# Transform settings
	trf = T.Compose([T.Resize(600),
					T.ToTensor(),
					T.Normalize(mean=[0.485, 0.456, 0.406],
								std=[0.229, 0.224, 0.225])])

	# Convert single image to single batch
	inp = trf(image).unsqueeze(0).to(device)
	style = trf(style_image).unsqueeze(0).to(device)
	target = inp.clone().to(device).requires_grad_(True)

	# General Model
	epochs = 500
	leaky = 1
	tanh = 0

	# Training loop
	a = [0.1, 0.15, 0.08, 0.00]
	# a = [0.3]
	for slope in a:

		model = models.vgg19(pretrained=True).features

		# Alter Architecture of pretrained model (only non-trainable layers)
		layers = []
		for name, layer in model.named_children():
			if isinstance(layer, nn.MaxPool2d):
				layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False))
			elif isinstance(layer, nn.ReLU) & (leaky == True):
				layers.append(nn.LeakyReLU(slope))
			elif isinstance(layer, nn.ReLU) & (tanh == True):
				layers.append(nn.Tanh())
			else:
				layers.append(layer)

		# Unpacking layers into new model
		model = nn.Sequential(*layers)
		print(model)

		# Making model not trainable
		for param in model.parameters():
			param.requires_grad = False

		# Grabbing outputs for activations
		activations = {}
		for name, layer in model.named_children():
			if isinstance(layer, nn.Conv2d):
				model[int(name)].register_forward_hook(extract_output(activations, name))

		# Sending model to GPU and defining optimizer
		model = model.to(device)
		optimizer = optim.Adam([target], lr=learning_rate)
		print(torch.cuda.memory_summary(device=None, abbreviated=False))

		for i in range(epochs):
			optimizer.zero_grad()
			torch.cuda.empty_cache()

			# Grab Target features from target image (image that will be altered over time as final output)
			c_target = features(target, model, content_layer)
			s_target = features(target, model, style_layers)

			# Obtaining features from original image and style image
			c_feature = features(inp, model, content_layer)
			s_features = features(style, model, style_layers)

			# Calculating content and style loss
			c_loss = content_loss(c_target, c_feature)
			s_loss = style_loss(s_target, s_features, s_weights)
			color_loss = color_l(target, style)

			# Calculating total loss
			total_loss = c_loss * c_weight + s_loss * s_weight + color_weight * color_loss

			# Backprop
			total_loss.backward()
			optimizer.step()

			if i % 10 == 0:
				print(f'Epoch {i}...')

		# activations histogram
		fig = plt.figure(figsize=(16, 9))
		fig.suptitle('Outputs from Activations')

		# Plot histogram for each activation layer
		n = len(activations.values())
		for i, hist in enumerate(activations):
			vals = activations[hist].squeeze(0).flatten().numpy()
			ax = fig.add_subplot(4, 4, i + 1) # n=16 so 4x4 grid
			ax.set_title(f'Layer {i}')
			ax.hist(vals, ec='white', color='black', bins=150)

		# Grabbing altered target image and converting it into an image to show
		fig2, ax2 = plt.subplots(1, 2, figsize=(16, 9))
		fig2.suptitle('Original vs. Stylized')

		# Converting Target tensor to image
		target_show = convert_to_image(target)

		ax2[1].imshow(target_show)
		ax2[1].set_title('Edited')

		# Converting Input tensor to image
		inp_show = convert_to_image(inp)
		ax2[0].imshow(inp_show)
		ax2[0].set_title('Original')

		# Saving figures
		fig.subplots_adjust(hspace=0.4, wspace=0.4)
		# fig.savefig(f'./results/activations_{slope}_lay.png')
		# fig2.savefig(f'./results/overall_{slope}_lay.png')

		del model
	# plt.show()

# ============================================================================
if __name__ == "__main__":
	main()

