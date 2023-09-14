#odigen-mlpmixer

This repository contains the official pytorch implementation of "Increasing diversity of omni-directional images
generated from single image using cGAN based on MLPMixer".

![GeneratorImage_v3.png](GeneratorImage_v3.png)

## Sample Images

![generated_image_sample.png](generated_image_sample.png)

## Model architecture

![model_architecture_v9.png](model_architecture_v9.png)

## Requirement

* Python 3.8 or avobe 
* PyTorch 1.10.2 or avobe
 
## Installation
  
To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

Additionally, make sure to install PyTorch from https://pytorch.org/get-started/locally/.

If you are using CUDA 11.3, you can install PyTorch with the following command:

```bash
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Dataset

The data can be downloaded from the following url
"add url here"

## Usage

To train the model, use the following command:

```bash
python train.py
```

## Evaluation

To evaluate the trained generator model, follow these steps:

1. Copy the trained generator to the eval/generators directory and rename it to "test.pth".
2. Run the following command from the root directory:

```bash
cd ./eval
python calc_all_metrics.py
```

The evaluation are stored in "eval/evaluation" folder.
