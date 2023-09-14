from os import listdir
from os.path import join
from PIL import Image
import glob
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import re
import os

class Sun360OutdoorDataset(data.Dataset):
	def __init__(self, image_dir, direction="b2a",train=True,transform=[]):
		super(Sun360OutdoorDataset, self).__init__()
		self.direction = direction
		path = join(join(image_dir,"train" if train else "test"), "base")
		print(path)
		try:
			self.mask = transforms.ToTensor()(Image.open("mask_img.png"))
		except:
			self.mask = transforms.ToTensor()(Image.open("../mask_img.png"))
		self.image_filenames = [x for x in glob.glob(path+"/*.jpg")]
		print("dataset size:",len(self.image_filenames))
		transform_list = transform + [transforms.ToTensor(),
						  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
		self.train = train

		self.transform = transforms.Compose(transform_list)

	def __getitem__(self, index):
		a = Image.open(self.image_filenames[index]).convert('RGB')
		a = self.transform(a)
		if self.train:
			a = torch.roll(a,tuple(np.random.randint(0,512,1)),dims=2)
		b = a * self.mask

		class_name_array = ["arena", "balcony", "beach", "boat_deck", "bridge", "cemetery",
							"coast", "desert", "field", "forest", "highway", "jetty", "lawn", "mountain", "park",
							"parking_lot", "patio", "plaza_courtyard", "ruin", "sports_field", "street",
							"swimming_pool", "train_station_or_track", "wharf"]

		class_label = torch.zeros(len(class_name_array))
		class_name = re.sub(r'_\d{,5}.jpg', '', os.path.basename(self.image_filenames[index]))
		class_label[class_name_array.index(class_name)] = 1

		if self.direction == "a2b":
			return a, b, class_label, self.image_filenames[index]
		else:
			return b, a, class_label, self.image_filenames[index]

	def __len__(self):
		return len(self.image_filenames)

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	set = Sun360OutdoorDataset("datasets/all_class_training/", train=True)
	print(len(set))
	for i in range(len(set)):
		a = set[i]
		plt.imshow(a[0].permute(1,2,0).numpy()/2+0.5)
		plt.show()
		break