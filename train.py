import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from losses import GANLossSoftPlus
from dataset import Sun360OutdoorDataset
from image_generate import generate_image
from tqdm import tqdm
import shutil
import mlflow
import fire
import time
from PIL import Image
import numpy as np

# settings of mlflow logging
mlflow_path = "./mlruns/"
tmp_path = "./tmp/"
mlflow.set_tracking_uri(mlflow_path)
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)
os.makedirs(tmp_path, exist_ok=True)

mask = torch.from_numpy(np.array(Image.open("mask_img.png"))).to(bool).reshape(1,1,256,512).cuda()

def cuda_setup(cuda, seed=0):
    torch.manual_seed(seed)
    if not cuda:
        return
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run --cuda=False")
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)


def save_model(net_g, net_d, path, epoch):
    net_g_model_out_path = os.path.join(path, f"netG_model_epoch_{epoch}.pth")
    net_d_model_out_path = os.path.join(path, f"netD_model_epoch_{epoch}.pth")
    torch.save(net_g, net_g_model_out_path)
    torch.save(net_d, net_d_model_out_path)
    mlflow.log_artifacts(path)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def train(
        batch_size=16,
        epoch_count=500,
        lr_g=1e-4,
        lr_d=1e-4,
        beta1=0.5,
        lamb=1,
        niter=400,
        cuda=True,
        g_channel=512,
        g_layer=4,
        g_tanh=True,
        g_kernel=3,
        g_single=False,
        update_rate=1,
        ndf=256,
        d_layer=8,
        d_channel_weight=0.01,
        g_channel_weight=0.1,
        seed=0,
        augment=True,
        experiment="MixerSphericalGAN",
        r1_gp=True,
        gamma = 0.001,
        weight_decay=0.00001):
    cuda_setup(cuda, seed)
    device = torch.device("cuda:0" if cuda else "cpu")
    g_layer = [g_layer for i in range(5)]

    mlflow.set_experiment(experiment)
    mlflow.start_run()

    params = {"batch_size": batch_size, "epoch_count": epoch_count,"lr_g": lr_g, "lr_d": lr_d,"beta1": beta1,
              "lamb": lamb,"update_rate":update_rate,"d_layer":d_layer,"g_kernel":g_kernel,"g_single":g_single,
              "ndf": ndf, "seed": seed, "niter": niter,"g_layer": g_layer,"g_channel_weight":g_channel_weight,
               "g_channel":g_channel, "gamma":gamma,"g_tanh":g_tanh,"d_channel_weight":d_channel_weight,
              "r1_gradient_penalty": r1_gp, "augment": augment, "weight_decay": weight_decay,}
    print(params)
    mlflow.log_params(params)

    print('Loading datasets')

    root_path = "datasets/"
    dataset = "all_class_training"


    train_set = Sun360OutdoorDataset(root_path + dataset, train=True)
    test_set = Sun360OutdoorDataset(root_path + dataset, train=False)

    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size,
                                      shuffle=True, drop_last=True, pin_memory=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=batch_size, pin_memory=True,
                                     shuffle=False)

    training_data_loader = sample_data(training_data_loader)

    print('Building models')
    net_g = Generator(g_channel, g_layer,g_tanh,g_kernel,g_single).cuda()
    net_d = Discriminator(6, ndf, layer=d_layer, augment=augment).cuda()

    criterionGAN_D =  GANLossSoftPlus()
    criterionGAN_G = GANLossSoftPlus()
    criterionL1 = nn.L1Loss().to(device)

    optimizer_g = optim.Adam(net_g.parameters(), lr=lr_g, betas=(beta1, 0.999), weight_decay=weight_decay)
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr_d, betas=(beta1, 0.999), weight_decay=weight_decay)

    start_time = time.time()

    epoch = 0

    for iteration in tqdm(range(niter*epoch_count), desc="Epoch"):
        if iteration % epoch_count == 0:
            loss_g_sum = 0
            loss_g_gan_l_sum = 0
            loss_g_gan_g_sum = 0
            loss_g_l1_sum = 0
            loss_d_sum = 0
            loss_d_real_l_sum = 0
            loss_d_real_g_sum = 0
            loss_d_fake_l_sum = 0
            loss_d_fake_g_sum = 0
            loss_d_real_l1_sum = 0
            loss_d_fake_l1_sum = 0
            grad_penalty_sum = 0
        net_g.eval()
        net_d.train()

        for i in range(update_rate):
            batch = next(training_data_loader)
            real_a, real_b, class_label = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # Generate Fake image for train discriminator
            with torch.no_grad():
                fake_b = net_g(real_a, class_label)

            # Update D network
            optimizer_d.zero_grad()

            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_l_fake, pred_g_fake, d_l1_fake_loss = net_d(fake_ab, class_label)
            loss_d_l_fake = criterionGAN_D(pred_l_fake, False) # adv patch loss
            loss_d_g_fake = criterionGAN_D(pred_g_fake, False) # adv channel loss

            # train with real
            if r1_gp:
                real_a.requires_grad = True
            real_ab = torch.cat((real_a, real_b), 1)

            pred_l_real, pred_g_real, d_l1_real_loss = net_d(real_ab, class_label)
            loss_d_l_real = criterionGAN_D(pred_l_real, True)
            loss_d_g_real = criterionGAN_D(pred_g_real, True)

            loss_d_real_l_sum += loss_d_l_real.item()
            loss_d_real_g_sum += loss_d_g_real.item()
            loss_d_fake_l_sum += loss_d_l_fake.item()
            loss_d_fake_g_sum += loss_d_g_fake.item()
            del pred_l_fake, pred_g_fake

            #calc gradient penalty
            grad_penalty = torch.tensor([0])
            if r1_gp:
                grad_real = torch.autograd.grad(outputs=pred_g_real.sum()+pred_l_real.sum(), inputs=real_a, create_graph=True)[0]
                grad_penalty = (grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = gamma * 10 / 2 * grad_penalty
                loss_d_l_real += grad_penalty
                grad_penalty_sum += grad_penalty.item()

            # Combined D loss
            loss_d = loss_d_l_fake + d_channel_weight * loss_d_g_fake + loss_d_l_real + d_channel_weight * loss_d_g_real + d_l1_fake_loss + d_l1_real_loss
            loss_d_sum += loss_d.item()
            loss_d_fake_l1_sum += d_l1_fake_loss.item()
            loss_d_real_l1_sum += d_l1_real_loss.item()

            loss_d.backward()

            del loss_d, loss_d_l_fake, loss_d_g_fake, loss_d_l_real, loss_d_g_real, d_l1_fake_loss, d_l1_real_loss,grad_penalty,grad_real
            optimizer_d.step()

        # train generator
        net_g.train()
        net_d.eval()

        optimizer_g.zero_grad()

        batch = next(training_data_loader)
        real_a, real_b, class_label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        fake_b = net_g(real_a, class_label)
        fake_ab = torch.cat((real_a, fake_b), 1)

        pred_fake_l, pred_fake_g, l1_ss = net_d(fake_ab, class_label)
        loss_g_gan_l = criterionGAN_G(pred_fake_l, True)
        loss_g_gan_g = criterionGAN_G(pred_fake_g, True)
        loss_g_l1 = criterionL1(mask * fake_b, mask * real_b) * lamb
        loss_g = loss_g_gan_l +  g_channel_weight * loss_g_gan_g + loss_g_l1 + l1_ss
        loss_g.backward()
        optimizer_g.step()
        loss_g_sum += loss_g.item()
        loss_g_gan_l_sum += loss_g_gan_l.item()
        loss_g_gan_g_sum += loss_g_gan_g.item()
        loss_g_l1_sum += loss_g_l1.item()
        del loss_g_l1, loss_g, loss_g_gan_l, loss_g_gan_g, l1_ss,fake_ab

        if (iteration + 1) % epoch_count == 0:
            epoch += 1
            mlflow.log_metrics({
                "loss_d": loss_d_sum / epoch_count / update_rate,
                "loss_d_real_local": loss_d_real_l_sum / epoch_count / update_rate,
                "loss_d_real_global": loss_d_real_g_sum / epoch_count / update_rate,
                "loss_d_fake_local": loss_d_fake_l_sum / epoch_count / update_rate,
                "loss_d_fake_channel": loss_d_fake_g_sum / epoch_count / update_rate,
                "loss_d_l1_real": loss_d_real_l1_sum / epoch_count / update_rate,
                "loss_d_l1_fake": loss_d_fake_l1_sum / epoch_count / update_rate,
                "loss_g": loss_g_sum / epoch_count,
                "loss_g_gan_local": loss_g_gan_l_sum / epoch_count,
                "loss_g_gan_channel": loss_g_gan_g_sum / epoch_count,
                "loss_g_l1": loss_g_l1_sum / epoch_count,
                "gradient_penalty": grad_penalty_sum / epoch_count / update_rate}, step=epoch)
            if epoch % 10 == 0:
                save_model(net_g, net_d, tmp_path, epoch)
            img = generate_image(testing_data_loader, net_g, require_classlabel=True)
            mlflow.log_image(img, f"epoch_{epoch:0>4}_test_preview.jpg")

    # save the latest net
    save_model(net_g, net_d, tmp_path, niter)

    now_time = time.time()
    t = now_time - start_time
    print(f"Training time: {t / 60:.1f}m")
    mlflow.set_tag("Training time minute", t / 60)
    shutil.rmtree(tmp_path)


if __name__ == '__main__':
    fire.Fire(train)