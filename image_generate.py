import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw, ImageFont

def generate_image(data_loader,generator,device="cuda",require_classlabel=True):
    dst = Image.new('RGB', (512 * 4, 256 * 4))
    draw = ImageDraw.Draw(dst)
    font = ImageFont.truetype("consolab.ttf", 24)
    n=0
    for batch in data_loader:
        input, _, class_label, test_file_name = batch[0].to(device), batch[1].to(device), batch[2].to(device),batch[3][0]
        with torch.no_grad():
            if require_classlabel:
                prediction = generator(input,class_label)
            else:
                prediction = generator(input)
        for i in prediction.detach():
            # make test preview

            out_img = i.cpu()
            image_numpy = out_img.float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            image_numpy = image_numpy.clip(0, 255)
            image_numpy = image_numpy.astype(np.uint8)
            image_pil = Image.fromarray(image_numpy)

            dst.paste(image_pil, ((n - 1) % 4 * 512, (n - 1) // 4 * 256))
            draw.text(((n - 1) % 4 * 512, (n - 1) // 4 * 256 + 1), test_file_name, (255, 255, 255), font=font)
            draw.text(((n - 1) % 4 * 512 - 1, (n - 1) // 4 * 256), test_file_name, (255, 255, 255), font=font)
            draw.text(((n - 1) % 4 * 512, (n - 1) // 4 * 256 - 1), test_file_name, (255, 255, 255), font=font)
            draw.text(((n - 1) % 4 * 512 + 1, (n - 1) // 4 * 256), test_file_name, (255, 255, 255), font=font)
            draw.text(((n - 1) % 4 * 512, (n - 1) // 4 * 256), test_file_name, (0, 0, 0), font=font)
            n += 1
        if n>16:
            break
    return dst



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dataset import Sun360OutdoorDataset
    test_set = Sun360OutdoorDataset("datasets/all_class_training")
    testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=5, shuffle=False)
    net = torch.nn.Identity().cuda()
    img = generate_image(testing_data_loader,net,require_classlabel=False)
    plt.imshow(img)
    plt.show()


