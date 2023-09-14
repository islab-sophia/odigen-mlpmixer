from dataset import Sun360OutdoorDataset
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import re
import os
import glob
import pandas as pd

def generate_images(model_path,output_path,model_name=None,use_predict_label=True):
    dataset = Sun360OutdoorDataset("../datasets/all_class_training", "a2b", train=False)

    dataloader = DataLoader(dataset,32,shuffle=False)
    print(len(dataset))
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
    model = torch.load(model_path).cuda()
    model.eval()

    if use_predict_label:
        try:
            df = pd.read_csv("predict.csv",index_col=0)
        except:
            df = pd.read_csv("eval/predict.csv",index_col=0)

    eye = torch.eye(24).cuda()

    with torch.inference_mode():
        for i in tqdm(dataloader):
            _,b,cls,file_name = i
            b = b.cuda()
            if use_predict_label:
                ls = [os.path.basename(file_name[j]) for j in range(len(file_name))]
                pred_cls = torch.from_numpy(df.loc[ls].values)

                pred_arg = torch.argmax(pred_cls, dim=1)
                cls = eye[pred_arg]
            else:
                cls = cls.cuda()
            img = model(b,cls)
            # img = K.normalize_min_max(img,0.,1.)
            img = torch.clip(img/2+0.5,0,1)
            for i in range(img.shape[0]):
                basename = os.path.basename(file_name[i])
                class_name = re.sub(r'_\d{,5}.jpg', '',basename)
                dir = os.path.join(output_path, model_name, class_name)
                os.makedirs(dir, exist_ok=True)
                torchvision.utils.save_image(img[i], os.path.join(dir, basename))

if __name__ == '__main__':
    ls = glob.glob("generators/*.pth")
    save_folder = "generated_img/"
    for i in ls:
        generate_images(i,save_folder)