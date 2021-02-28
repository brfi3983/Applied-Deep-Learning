<!-- #####
# 1) Choose Deeper Style Layers
# 2) Choose / Add Deeper Content Layers
# 3) Analysis of Activation functions for photorealistic outputs
# 4) Faster computation for batch editing?
# 5) HyperParameters/Architecture?
# 6) Visualization/Analysis? -->

<!-- ##### Extracting intermediate architectures/layers!
# x = nn.Sequential(*list(model.feature_extractor.classifier.children()))
# print(x)
# test = []
# for num, layer in x.named_children():
# 	test.append(layer)
# print(test)
# y = nn.Sequential(*test)
# print(y)
	# if isinstance(layer, torch.nn.ReLU):
	# 	print(layer)
# print(model.feature_extractor.classifier, x)
# for name, layer in model.named_modules():
# 	if isinstance(layer, torch.nn.Conv2d):
# 		print(name) -->