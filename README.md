# has-8
[Hybrid Layer-Wise ANN-SNN With Surrogate Spike Encoding-Decoding Structure](https://arxiv.org/abs/2509.24411)

## Quickstart

This is a quickstart guide on how to use our model as a package 

### Step 1: Environment setup and repo download

To setup the environment testing with this encoder, you will need Pytorch, Pennylane and SpikingJelly. We suggest using conda environment with:

```bash
$ conda create -n env python=3.12.2
$ conda install pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia #As latest pytorch conda guide, change cuda version suitable to your case.
$ pip install spikingjelly
$ pip install git+https://github.com/luutn2002/has-8.git
```

or clone and modify locally:

```bash
$ git clone https://github.com/luutn2002/has-8.git
$ cd has-8
$ pip install -r requirements.txt
```

### Step 2: Usage

To ensure reproducibility, remember to use static random seed:
```python
import torch
import numpy as np
import random

seed = 3407

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

To use the HAS-8-VGG model, we can simply import as a normal Pytorch model:
```python
from has_8 import has8_vgg_b16_m2_d4, has8_vgg_b24_m2_d4

IN_CHANNELS = 3
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = has8_vgg_b16_m2_d4(in_channels=IN_CHANNELS, 
                           out_channels=NUM_CLASSES).to(DEVICE)
# Or
model = has8_vgg_b24_m2_d4(input_channels=IN_CHANNELS, 
                           out_channels=NUM_CLASSES).to(DEVICE)
```
For HAS-8-ResNet:
```python
from has_8 import has8_rn_b32_m2_d4, has8_rn_b64_m2_d4

IN_CHANNELS = 3
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = has8_rn_b32_m2_d4(input_channels=IN_CHANNELS, 
                          num_classes=NUM_CLASSES).to(DEVICE)
# Or
model = has8_rn_b64_m2_d4(input_channels=IN_CHANNELS, 
                          num_classes=NUM_CLASSES).to(DEVICE)
```
## Preprocessing
All used dataset and transform is included in [torchvision](https://docs.pytorch.org/vision/main/datasets.html). Preprocess with torch for images larger than 224x224:
```python
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM_TRAIN = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
])

TRANSFORM_TEST = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
])

train_set = ...

val_set = ...

train_loader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True,
                                            #generator=torch.Generator(device=DEVICE),
                                            collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in torch.utils.data.dataloader.default_collate(x)))
test_loader = torch.utils.data.DataLoader(val_set, 
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            #generator=torch.Generator(device=DEVICE),
                                            collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in torch.utils.data.dataloader.default_collate(x)))
```                                  
## Optimizer settings

For Adam, used in small scale datasets:
```python
optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-3, 
                                 weight_decay=1e-3)
```
For SGDR, used in ImageNet:
```python
BATCH_SIZE = 16
EPOCHS = 100

base_lr = 0.1*(BATCH_SIZE/256)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=base_lr,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=base_lr/10, total_iters=5)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 5)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
```
## License

Source code is licensed under MIT License.

## Contribution guidelines

Please open an issue or pull request if there are bugs or contribution to be made. Thank you.

## Citations
Paper is under review. Temporarily please cite as:
```bibtex
@article{luu2025hybrid,
  title={Hybrid Layer-Wise ANN-SNN With Surrogate Spike Encoding-Decoding Structure},
  author={Luu, Nhan T and Luu, Duong T and Pham, Nam N and Truong, Thang C},
  journal={arXiv preprint arXiv:2509.24411},
  year={2025}
}
```
