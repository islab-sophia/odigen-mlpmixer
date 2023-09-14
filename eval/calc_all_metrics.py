import os
import pandas as pd
try:
    import openpyxl
except:
    print("openpyxl is not exists.")
    print("please install openpyxl.")
    print("pip install openpyxl")
    exit(-1)

from scene_recognition_extract10_GPU import scene_recog
from inceptionv3_extract10_phi_FID_GPU import calc_fid
from inceptionv3_extract10_phi_IS_GPU import calc_is
from evaluate_continuites_equi import calc_cont
from image_generate import generate_images
from extract_multi_views import generate_multi_views


def main(target_folder="cbn"):
    os.makedirs("./evaluation",exist_ok=True)
    print(target_folder)
    sph = os.path.join("generated_img",target_folder)
    ext = os.path.join("extracted_phi",target_folder)
    recog = scene_recog(ext,True)
    fid = calc_fid(ext,True)
    is_ = calc_is(ext,True)
    cont = calc_cont(sph,True)
    fid_is = pd.concat([fid,is_],axis=0)
    with pd.ExcelWriter(f"./evaluation/{target_folder}.xlsx") as writer:
        fid_is.to_excel(writer, sheet_name="FID_IS", index=True, header=True)
        recog.to_excel(writer, sheet_name="Scene Recognition", index=True, header=True)
        cont.to_excel(writer, sheet_name="continuities", index=False, header=True)



def main_roop(*args):
    print(args)
    if len(args)==0:
        main()
    else:
        for i in args:
            main(i)

if __name__ == '__main__':
    import glob
    import os
    ls = glob.glob("generators/*.pth")
    save_folder = "generated_img/"
    output = "extracted_phi/"
    print(ls)
    for i in ls:
        generate_images(i, save_folder)
    folder_name_array = [os.path.join(save_folder, i) for i in os.listdir(save_folder)]
    for folder_name in folder_name_array:
        generate_multi_views(folder_name, output)
    ls = [os.path.splitext(os.path.basename(i))[0] for i in ls]
    main_roop(*ls)