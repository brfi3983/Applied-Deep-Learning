import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
plt.style.use('ggplot')

def ReLU(x):
	return np.maximum(0, x)
def LeakyReLU(x, a):
	return np.maximum(a*x, x)
# ========================================================
def main():

	# Importing Data
	image = np.asarray(Image.open('Dolomites.jpg'))
	style_image = np.asarray(Image.open('shortstache_style.jpg'))
	print(f'Shape of Original: {image.shape}, Shape of Style: {style_image.shape}')

	# Raw Images
	plt.figure(figsize=(16,9))
	plt.suptitle('Original vs. Style', fontsize=30)
	plt.subplot(1, 2, 1)
	plt.title('Original Image')
	plt.imshow(image)
	plt.subplot(1, 2, 2)
	plt.title('Style Image')
	plt.imshow(style_image)

	# Histograms
	# Light
	plt.figure(figsize=(16,9))
	plt.suptitle('Luminance Histograms', fontsize=30)
	plt.subplot(1, 2, 1)
	plt.title('Original')
	plt.hist(image.ravel(), 256, [0, 256], color='grey')
	plt.subplot(1, 2, 2)
	plt.title('Style')
	plt.hist(style_image.ravel(), 256, [0, 256], color='grey')

	# Color
	plt.figure(figsize=(16,9))
	plt.suptitle('Color Histograms', fontsize=30)
	plt.subplot(1, 2, 1)
	plt.title('Original')
	plt.hist(image[:,:,0].ravel(), density=True, bins=256, histtype='step', color='red')
	plt.hist(image[:,:,1].ravel(), density=True, bins=256, histtype='step', color='green')
	plt.hist(image[:,:,2].ravel(), density=True, bins=256, histtype='step', color='blue')
	plt.subplot(1, 2, 2)
	plt.title('Style')
	plt.hist(style_image[:,:,0].ravel(), density=True, bins=256, histtype='step', color='red')
	plt.hist(style_image[:,:,1].ravel(), density=True, bins=256, histtype='step', color='green')
	plt.hist(style_image[:,:,2].ravel(), density=True, bins=256, histtype='step', color='blue')


	# Activation Functions
	plt.figure(figsize=(16,9))
	x = np.linspace(-3, 3, 50)
	a = [0.3, 0.15, 0.08]
	y_leaky_relu = {}
	y_relu = ReLU(x)
	for val in a:
		y_leaky_relu[str(val)] = LeakyReLU(x, val)

	plt.title('ReLU, Tanh, and Leaky ReLU', fontsize=30)
	plt.plot(x, y_relu, ls=':', label='ReLU')
	plt.plot(x, np.tanh(x), ls='--', label='Tanh')
	plt.plot(x, y_leaky_relu['0.3'], label='LeakyReLU with a = 0.3')
	plt.plot(x, y_leaky_relu['0.15'], label='LeakyReLU with a = 0.15')
	plt.plot(x, y_leaky_relu['0.08'], label='LeakyReLU with a = 0.08')

	plt.legend()

	plt.show()
# ========================================================
if __name__ == "__main__":
	main()