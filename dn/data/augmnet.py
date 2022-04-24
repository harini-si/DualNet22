import random

from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

MEAN, STD = (0.082811184, 0.22163138)


class Solarization(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			return ImageOps.solarize(img)
		else:
			return img


VCTransform = transforms.Normalize(mean=[MEAN], std=[STD])


class BarlowAugment:
	def __init__(self):
		self.transform = transforms.Compose(
				[
						transforms.RandomResizedCrop(20),
						transforms.RandomHorizontalFlip(p=0.5),
						transforms.GaussianBlur((3, 3), (0.1, 2.0)),
						transforms.RandomSolarize(threshold=0.2, p=0.0),
						transforms.Normalize(mean=[MEAN], std=[STD]),
				]
		)

		self.transform_prime = transforms.Compose(
				[
						transforms.RandomResizedCrop(20),
						transforms.RandomHorizontalFlip(p=0.5),
						transforms.GaussianBlur((3, 3), (0.1, 2.0)),
						transforms.RandomSolarize(threshold=0.5, p=0.0),
						transforms.Normalize(mean=[MEAN], std=[STD]),
				]
		)

	def __call__(self, x):
		y1 = self.transform(x)
		# y1 = x
		y2 = self.transform_prime(x)
		return y1, y2
