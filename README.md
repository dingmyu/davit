# DaViT: Dual Attention Vision Transformer

This repo contains the official detection and segmentation implementation of paper "[DaViT: Dual Attention Vision Transformer](https://arxiv.org/pdf/2204.03645.pdf)", by Mingyu Ding, Bin Xiao, Noel Codella, Ping Luo, Jingdong Wang, and Lu Yuan.

### The official implementation for image classification will be released in [https://github.com/microsoft/DaViT](https://github.com/microsoft/DaViT).


## Getting Started
Python3, PyTorch>=1.8.0, torchvision>=0.7.0 are required for the current codebase.

```shell
# An example on CUDA 10.2
pip install torch===1.9.0+cu102 torchvision===0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install thop pyyaml fvcore pillow==8.3.2
```

### Object Detection and Instance Segmentation
- `cd mmdet` & install mmcv/mmdet
  ```shell
  # An example on CUDA 10.2 and pytorch 1.9
  pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
  pip install -r requirements/build.txt
  pip install -v -e .  # or "python setup.py develop"
  ```

- `mkdir data` & Prepare the dataset in data/coco/ (Format: ROOT/mmdet/data/coco/annotations, train2017, val2017)
  
- Finetune on COCO
  ```shell
  bash tools/dist_train.sh configs/davit_retinanet_1x_coco.py 8 \
  --cfg-options model.pretrained=PRETRAINED_MODEL_PATH
  ```

### Semantic Segmentation
- `cd mmseg` & install mmcv/mmseg
  ```shell
  # An example on CUDA 10.2 and pytorch 1.9
  pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
  pip install -e .
  ```
  
- `mkdir data` & Prepare the dataset in data/ade/ (Format: ROOT/mmseg/data/ADEChallengeData2016)
  
- Finetune on ADE 
  ```shell
  bash tools/dist_train.sh configs/upernet_davit_512x512_160k_ade20k.py 8 \
  --options model.pretrained=PRETRAINED_MODEL_PATH
  ```

- Multi-scale Testing
  ```shell
  bash tools/dist_test.sh configs/upernet_davit_512x512_160k_ade20k.py \ 
  TRAINED_MODEL_PATH 8 --aug-test --eval mIoU
  ```

## Benchmarking

### Image Classification on [ImageNet-1K](https://www.image-net.org/)

| Model | Pretrain | Resolution | acc@1 | acc@5 | #params | FLOPs | Checkpoint | Log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| DaViT-T | IN-1K | 224 | 82.8 | 96.2 | 28.3M   | 4.5G   | [download](https://drive.google.com/file/d/1RSpi3lxKaloOL5-or20HuG975tbPwxRZ/view?usp=sharing) | [log](https://drive.google.com/file/d/1JL0IoSYdcCG6lGnMAlWxJmiu1mY4c99k/view?usp=sharing) |
| DaViT-S | IN-1K | 224 | 84.2 | 96.9 | 49.7M   | 8.8G   | [download](https://drive.google.com/file/d/1q976ruj45mt0RhO9oxhOo6EP_cmj4ahQ/view?usp=sharing) | [log](https://drive.google.com/file/d/1u8gCY8wvrz1wlYLUhpg0YN6KSFk2UYob/view?usp=sharing) |
| DaViT-B | IN-1K | 224 | 84.6 | 96.9 | 87.9M   | 15.5G  | [download](https://drive.google.com/file/d/1u9sDBEueB-YFuLigvcwf4b2YyA4MIVsZ/view?usp=sharing) | [log](https://drive.google.com/file/d/1gEWbT5uj8dHY0CAoTFueXIt0M9F0OHZ8/view?usp=sharing) |

### Object Detection and Instance Segmentation on [COCO](https://cocodataset.org/#home)

#### [Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)

| Backbone | Pretrain | Lr Schd | #params | FLOPs | box mAP | mask mAP | Checkpoint | Log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DaViT-T | ImageNet-1K | 1x | 47.8M | 263G | 45.0 | 41.1 | [download](https://drive.google.com/file/d/1kXVQdXmcBN6FNfgL3BO5APNRGHQFZwzB/view?usp=sharing) | [log](https://drive.google.com/file/d/1d6nAd6YBPrDK2eg1-Or7XkUHUJgpkDcG/view?usp=sharing) |
| DaViT-T | ImageNet-1K | 3x | 47.8M | 263G | 47.4 | 42.9 | [download](https://drive.google.com/file/d/1CfqaZ5xzVuK3EFeslwI_RS4sIY6sh6aK/view?usp=sharing) | [log](https://drive.google.com/file/d/1utz3aHVZ7gnnSh3fytAGc8Q195xN2PAw/view?usp=sharing) |
| DaViT-S | ImageNet-1K | 1x | 69.2M | 351G | 47.7 | 42.9 | [download](https://drive.google.com/file/d/1psE7P775kmniHCKFU83gHfpU9skwQ2Jo/view?usp=sharing) | [log](https://drive.google.com/file/d/1MXd6U3UIdcmToNplIK78umnGMLhsVW7I/view?usp=sharing) |
| DaViT-S | ImageNet-1K | 3x | 69.2M | 351G | 49.5 | 44.3 | [download](https://drive.google.com/file/d/1INHzXjLynO5eelmg8f_bgzwylVZUbpI5/view?usp=sharing) | [log](https://drive.google.com/file/d/1tcSwY4ie2nY0VxfGpDT7AHJaG6eX7qR6/view?usp=sharing) |
| DaViT-B | ImageNet-1K | 1x | 107.3M | 491G | 48.2 | 43.3 | [download](https://drive.google.com/file/d/1HyBF8WapjOI78_2U45iOsdJ8zI41sBL4/view?usp=sharing) | [log](https://drive.google.com/file/d/1FfDzOKaZGagH-u7NHy8i2ObOc7xYllcp/view?usp=sharing) |
| DaViT-B | ImageNet-1K | 3x | 107.3M | 491G | 49.9 | 44.6 | [download](https://drive.google.com/file/d/16HTUwAxm3uKXhxzRJADYODDCz-Co_zvM/view?usp=sharing) | [log](https://drive.google.com/file/d/1_dsx83yxGLWFOLhpp9KFU8rJPUUaKjMj/view?usp=sharing) |

#### [RetinaNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

| Backbone | Pretrain | Lr Schd | #params | FLOPs | box mAP | Checkpoint | Log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DaViT-T | ImageNet-1K | 1x | 38.5M | 244G | 44.0 | [download](https://drive.google.com/file/d/1pmmCgek4opgLu4Q3D3rBilgVaCDYGTmi/view?usp=sharing) | [log](https://drive.google.com/file/d/1zcjyfKZPLxl3ADacH5RjXOYGCRF-JVaf/view?usp=sharing) |
| DaViT-T | ImageNet-1K | 3x | 38.5M | 244G | 46.5 | [download](https://drive.google.com/file/d/10reUbHDePMFeQODvAHn856Jjd1eXQgHC/view?usp=sharing) | [log](https://drive.google.com/file/d/1CvLhFqgEHrz_nlplHdUy_4wngtuPuLr8/view?usp=sharing) |
| DaViT-S | ImageNet-1K | 1x | 59.9M | 332G | 46.0 | [download](https://drive.google.com/file/d/1weqNFG5BIJjSPXqEDiGIOzcdxZYGwjcg/view?usp=sharing) | [log](https://drive.google.com/file/d/1n05r49Ggx99MwlcACywGtzAN1bIMotZ9/view?usp=sharing) |
| DaViT-S | ImageNet-1K | 3x | 59.9M | 332G | 48.2 | [download](https://drive.google.com/file/d/1olT5sSTm9cdsHPILtxCJNW_68iVUcU85/view?usp=sharing) | [log](https://drive.google.com/file/d/1nDHNW0uNfDwJGniyxBAf0XuiEQKsx1yr/view?usp=sharing) |
| DaViT-B | ImageNet-1K | 1x | 98.5M | 471G | 46.7 | [download](https://drive.google.com/file/d/14KNYGr_z5c1ZgJXOV_yVu96Uo8Iu7OjR/view?usp=sharing) | [log](https://drive.google.com/file/d/1PoGwMYsj3poEhLMaiIE2B2Vhh8XQULBc/view?usp=sharing) |
| DaViT-B | ImageNet-1K | 3x | 98.5M | 471G | 48.7 | [download](https://drive.google.com/file/d/12dURuwqMMtU_A8SGS47M1Q4QsjN8L1uk/view?usp=sharing) | [log](https://drive.google.com/file/d/1d6nAd6YBPrDK2eg1-Or7XkUHUJgpkDcG/view?usp=sharing) |

### Semantic Segmentation on [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

| Backbone | Pretrain  | Method | Resolution | Iters | #params | FLOPs | mIoU | Checkpoint | Log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DaViT-T | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 60M  | 940G | 46.3 | [download](https://drive.google.com/file/d/1UHrdthX3N30YkQZz7eBq0gO2uLCYP3r_/view?usp=sharing) | [log](https://drive.google.com/file/d/1VNs7CrKghhl7c-DRx-l8tkTAdVCzABF-/view?usp=sharing) |
| DaViT-S | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 81M | 1030G | 48.8 | [download](https://drive.google.com/file/d/1NYR9axwsXR-GccpLWzrGSRv-L3TzDy99/view?usp=sharing) | [log](https://drive.google.com/file/d/1iF36MqZvayBL41tJIo_2A35OCP9uXlhs/view?usp=sharing) |
| DaViT-B | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 121M | 1175G | 49.4 | [download](https://drive.google.com/file/d/1-3berKLlmg01IVYUr-3gtcfQ5yZLiO7_/view?usp=sharing) | [log](https://drive.google.com/file/d/1MSMLYbmX2bN9NlNXJmMku0dKVnwKzJPY/view?usp=sharing) |


## Citation

If you find this repo useful to your project, please consider citing it with following bib:

    @article{ding2022davit,
        title={DaViT: Dual Attention Vision Transformer}, 
        author={Ding, Mingyu and Xiao, Bin and Codella, Noel and Luo, Ping and Wang, Jingdong and Yuan, Lu},
        journal={arXiv preprint arXiv:2204.03645},
        year={2022},
    }

## Acknowledgement

Our codebase is built based on [timm](https://github.com/rwightman/pytorch-image-models), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We thank the authors for the nicely organized code!
