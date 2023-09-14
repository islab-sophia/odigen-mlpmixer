import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from scipy import linalg
from torchvision import transforms as trn
from torchvision.models.inception import inception_v3
from tqdm import tqdm

def normalize(v, axis=-1, order=2):
	l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
	l2[l2==0] = 1
	return v/l2


class PartialInceptionNetwork(nn.Module):
    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def calc_fid(target_folder="extracted_phi/cbn",return_dataframe=False):
	"""
	生成全天球画像から様々な仰角で抽出したスナップショット画像について、FIDを求める
	"""
	phi_cs_array = [90, 45, 0, -45, -90]
	assert torch.cuda.is_available()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	x = 200
	y = 150
	crop = 150
	print(f'mode:{device}')
	print("calculating FID")
	print(target_folder)

	target_results_array = []

	inception_model = PartialInceptionNetwork()
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


		feature_map_array = torch.empty((0, 2048)).to(device)
		directory = os.path.join(target_folder,f"phi{phi}")
		dataset = torchvision.datasets.ImageFolder(directory,transform=centre_crop)
		dataloader = data.DataLoader(dataset, 64,num_workers=2)

		for img,_ in tqdm(dataloader):
			img = img.to(device)
			with torch.inference_mode():
				logit = inception_model(img)

			feature_map_array = torch.cat([feature_map_array, logit], dim=0)

		feature_map_array = feature_map_array.cpu().numpy().astype(np.float64)


		"""
		mu = np.mean(feature_map_array, axis=0)
		np.save(f'npy/test_extract10_phi{phi}_mu_inceptionv3.npy', mu)

		sigma = np.cov(feature_map_array, rowvar=False)
		np.save(f'npy/test_extract10_phi{phi}_sigma_inceptionv3.npy', sigma)
		exit()
		"""
		try:
			_mu1 = np.load(f'npy/test_extract10_phi{phi}_mu_inceptionv3.npy')
			_sigma1 = np.load(f'npy/test_extract10_phi{phi}_sigma_inceptionv3.npy')
		except:
			_mu1 = np.load(f'eval/npy/test_extract10_phi{phi}_mu_inceptionv3.npy')
			_sigma1 = np.load(f'eval/npy/test_extract10_phi{phi}_sigma_inceptionv3.npy')

		mu1 = np.atleast_1d(_mu1)
		sigma1 = np.atleast_2d(_sigma1)

		mu2 = np.atleast_1d(np.mean(feature_map_array, axis=0))
		sigma2 = np.atleast_2d(np.cov(feature_map_array, rowvar=False))

		eps=1e-6

		assert mu1.shape == mu2.shape, \
			'Training and test mean vectors have different lengths'
		assert sigma1.shape == sigma2.shape, \
			'Training and test covariances have different dimensions'

		diff = mu1 - mu2

		# Product might be almost singular
		covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
		if not np.isfinite(covmean).all():
			msg = ('fid calculation produces singular product; '
					'adding %s to diagonal of cov estimates') % eps
			print(msg)
			offset = np.eye(sigma1.shape[0]) * eps
			covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

		# Numerical error might give slight imaginary component
		if np.iscomplexobj(covmean):
			if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
				m = np.max(np.abs(covmean.imag))
				raise ValueError('Imaginary component {}'.format(m))
			covmean = covmean.real

		tr_covmean = np.trace(covmean)

		result_array += [diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean]
		target_results_array += [sum(result_array)/len(result_array)]

	del inception_model,img

	df = pd.DataFrame(target_results_array, index=phi_cs_array,columns=["FID"]).round(4).T
	print(df)
	if return_dataframe:
		return df
	else:
		save_folder_name = os.path.split(target_folder)
		if save_folder_name[-1] == "":
			save_folder_name = os.path.split(save_folder_name[0])
		df.to_csv(f"{save_folder_name[-1]}_FID.csv", )


if __name__ == '__main__':
	import fire
	fire.Fire(calc_fid)
