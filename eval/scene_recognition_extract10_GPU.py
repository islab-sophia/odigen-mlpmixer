import os.path

import pandas as pd
import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def scene_recog(target_folder="../extracted_phi/generated_img/cbn",return_dataframe=False):
	"""
	全天球画像から抽出したスナップショット画像のシーン認識評価プログラム
	仰角は5方向[90°, 45°, 0°, -45°, -90°]、水平方向は10方向[0～360°まで36°ずつ]に抽出
	同じ入力テスト画像に対して5回生成を繰り返したものについて平均を求める(変数rep_numを参照)
	仰角ごとに抽出画像の保存フォルダを分ける(変数terget_dirを参照)
	"""
	phi_cs_array = [90, 45, 0, -45, -90]
	assert torch.cuda.is_available()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_of_classes = 24
	arch = 'resnet18'
	x = 200
	y = 150
	crop = 150
	print("calculating scene recognition")
	print(f'mode:{device}')

	# load the image transformer
	centre_crop = trn.Compose([
			trn.Resize((x,y)),
			trn.CenterCrop(crop),
			trn.ToTensor(),
			trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	print(target_folder)

	df_ls = []
	for phi in tqdm(phi_cs_array, desc="Phi"):

		# 重みファイルの読み込み，ネットワークの設定---------------------------------------------
		model_file = f'models/nomal_for_fine_tuning_t9v1_resnet18_max50_phi{phi}.pth.tar'


		# -----------------------------------------------------------------------------

		model = models.__dict__[arch](num_classes = n_of_classes)
		try:
			checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
		except:
			checkpoint = torch.load("eval/"+model_file, map_location=lambda storage, loc: storage)

		state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict)
		model.eval()
		model.to(device)

		directory = os.path.join(target_folder,f"phi{phi}")
		dataset = ImageFolder(directory,centre_crop)
		dataloader = DataLoader(dataset,64)
		label_true = torch.zeros((0,))
		label_pred = torch.zeros((0,24)).cuda()

		with torch.inference_mode():
			for img,label in tqdm(dataloader):
				label_true = torch.cat([label_true,label],0)
				img = img.to(device)
				pred = model(img)
				h_x = F.softmax(pred,1)
				label_pred = torch.cat([label_pred,h_x],0)

		label_true = label_true.numpy().astype(int)
		label_pred = label_pred.cpu().numpy()
		classes = dataset.classes
		accuracy = np.mean(label_true == np.argmax(label_pred,axis=1))
		# pred_rate = np.mean(label_pred[label_true])

		accuracy_class = []
		# pred_rate_class = []
		for i in range(24):
			accuracy_class.append(np.mean(np.argmax(label_pred[label_true==i],axis=1)==i))
			# pred_rate_class.append(np.mean(label_pred[label_true==i,i]))

		classes.append("average")
		accuracy_class.append(accuracy)
		# pred_rate_class.append(pred_rate)
		# table = np.concatenate([np.array(accuracy_class).reshape(-1,1),np.array(pred_rate_class).reshape(-1,1)],axis=1)
		df = pd.DataFrame(accuracy_class,index=classes)
		df_ls.append(df)

	del model, img

	df = pd.concat(df_ls,axis=1)
	df.columns = phi_cs_array
	print(df)
	if return_dataframe:
		return df
	else:
		save_folder_name = os.path.split(target_folder)
		if save_folder_name[-1] == "":
			save_folder_name = os.path.split(save_folder_name[0])
		df.to_csv(f"{save_folder_name[-1]}_scene_recog.csv", )

if __name__ == '__main__':
	import fire
	fire.Fire(scene_recog)

