import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision
from scipy.stats import entropy
from torch.nn import functional as F
from torchvision import transforms as trn
from torchvision.models.inception import inception_v3
from tqdm import tqdm


def normalize(v, axis=-1, order=2):
	l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
	l2[l2==0] = 1
	return v/l2


def calc_is(target_folder="extracted_phi/cbn",return_dataframe=False):
	"""
	生成全天球画像から様々な仰角で抽出したスナップショット画像について、ISを求める
	"""
	phi_cs_array = [90, 45, 0, -45, -90]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	x = 200
	y = 150
	crop = 150
	splits = 1
	print(f'mode:{device}')
	print("calculating IS")
	print(target_folder)

	target_results_array = []

	dtype = torch.cuda.FloatTensor
	inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
	inception_model.eval()
	inception_model.to(device)

	# load the image transformer
	centre_crop = trn.Compose([
			trn.Resize((x,y)),
			trn.CenterCrop(crop),
			trn.Resize((299,299)),
			trn.ToTensor(),
			trn.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	for phi in tqdm(phi_cs_array, desc="Phi"):

		result_array = []
		feature_map_array = torch.empty((0, 1000)).to(device)
		directory = os.path.join(target_folder,f"phi{phi}")
		dataset = torchvision.datasets.ImageFolder(directory,transform=centre_crop)
		dataloader = data.DataLoader(dataset, 64,num_workers=2)

		for img,_ in tqdm(dataloader):
			img = img.to(device)
			with torch.inference_mode():
				logit = inception_model(img)

			feature_map_array = torch.cat([feature_map_array, logit], dim=0)

		feature_map_array = F.softmax(feature_map_array, dim=1).cpu().numpy().astype(np.float64)
		N = feature_map_array.shape[0]-1
		# KL-div
		split_scores = []
		for k in range(splits):
			part = feature_map_array[k * (N // splits): (k+1) * (N // splits), :]
			py = np.mean(part, axis=0)
			scores = []
			for i in range(part.shape[0]):
				pyx = part[i, :]
				scores.append(entropy(pyx, py))
			split_scores.append(np.exp(np.mean(scores)))

		# np.mean(split_scores), np.std(split_scores)
		result_array += [np.mean(split_scores)]
		target_results_array += [sum(result_array)/len(result_array)]

	del inception_model, img

	df = pd.DataFrame(target_results_array, index=phi_cs_array, columns=["IS"]).round(4).T
	print(df)
	if return_dataframe:
		return df
	else:
		save_folder_name = os.path.split(target_folder)
		if save_folder_name[-1] == "":
			save_folder_name = os.path.split(save_folder_name[0])
		df.to_csv(f"{save_folder_name[-1]}_IS.csv", )


if __name__ == '__main__':
	import fire
	fire.Fire(calc_is)
