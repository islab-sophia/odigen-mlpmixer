#!/usr/bin/env python
# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import os

import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

def calc_cont(target_folder="../generated_img/cbn",return_dataframe=False):
	"""
	生成した正距円筒図法での全天球画像の連続性の評価を行う
	T(Top)↓：上辺の1行分のピクセルが同じピクセル値になっているかを標準偏差を用いて評価
	B(Bottom)↓：下辺の1行分のピクセルが同じピクセル値になっているかを標準偏差を用いて評価
	S(Side)↓：左辺と右辺の連続しているピクセルの値差を各ピクセルそれぞれ算出し、平均二乗平方根で評価
	"""
	# 設定---------------------------------------------
	# -----------------------------------------------------------------------------
	
	class_name_array = ["arena","balcony","beach","boat_deck","bridge","cemetery","coast","desert","field","forest",
	"highway","jetty","lawn","mountain","park","parking_lot","patio","plaza_courtyard","ruin","sports_field",
	"street","swimming_pool","train_station_or_track","wharf"]

	difT_array = []
	difB_array = []
	difS_array = []


	print("calculating continuites")
	print(target_folder)


	for class_name in tqdm(class_name_array):
		dir = target_folder + f"/{class_name}/"
		files = os.listdir(dir)

		for file in tqdm(files):
			img = np.array(Image.open(dir + file))

			h, w, c = img.shape

			# Top
			imgT = img[0, :, :]
			stdT = np.std(imgT, axis=0)
			difT_array.append(np.mean(stdT))

			# Bottom
			imgB = img[h-1, :, :]
			stdB = np.std(imgB, axis=0)
			difB_array.append(np.mean(stdB))

			# Side
			imgL = img[:, 0, :]
			imgR = img[:, w-1, :]
			difS_array.append(np.sqrt(np.mean((imgL-imgR)**2)))

	csv_file_name = f"evaluation_equi/{target_folder}_test.csv"
	df = pd.DataFrame([np.mean(difT_array),np.mean(difB_array),np.mean(difS_array)],
					  index=["Top","Bottom","Side"]).round(4).T
	if return_dataframe:
		return df
	else:
		df.to_csv(csv_file_name,index=False)


if __name__ == '__main__':
	import fire
	fire.Fire(calc_cont)