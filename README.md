# DaViT: Dual Attention Vision Transformer (ECCV 2022)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/davit-dual-attention-vision-transformers/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=davit-dual-attention-vision-transformers)

This repo contains the official detection and segmentation implementation of paper "[DaViT: Dual Attention Vision Transformer (ECCV 2022)](https://arxiv.org/pdf/2204.03645.pdf)", by Mingyu Ding, Bin Xiao, Noel Codella, Ping Luo, Jingdong Wang, and Lu Yuan. See [Introduction.md](./Introduction.md) for an introduction.

**The large models for image classification will be released in [https://github.com/microsoft/DaViT](https://github.com/microsoft/DaViT).**

## Introduction

![teaser](figures/teaser.png)

In this work, we introduce Dual Attention Vision Transformers (DaViT), a simple yet effective vision transformer architecture that is able to capture global context while maintaining computational efficiency.
We propose approaching the problem from an orthogonal angle: exploiting self-attention mechanisms with both "spatial tokens" and "channel tokens".
(i) Since each channel token contains an abstract representation of the entire image, the channel attention naturally captures global interactions and representations by taking all spatial positions into account when computing attention scores between channels.
(ii) The spatial attention refines the local representations by performing fine-grained interactions across spatial locations, which in turn helps the global information modeling in channel attention.

![architecture](figures/architecture.png)

Experiments show our DaViT achieves state-of-the-art performance on four different tasks with efficient computations.
Without extra data, DaViT-Tiny, DaViT-Small, and DaViT-Base achieve 82.8%, 84.2%, and 84.6% top-1 accuracy on ImageNet-1K with 28.3M, 49.7M, and 87.9M parameters, respectively.
When we further scale up DaViT with 1.5B weakly supervised image and text pairs, DaViT-Gaint reaches 90.4% top-1 accuracy on ImageNet-1K.

![acc](figures/acc.png)

## Getting Started
Python3, PyTorch>=1.8.0, torchvision>=0.7.0 are required for the current codebase.

```shell
# An example on CUDA 10.2
pip install torch===1.9.0+cu102 torchvision===0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install thop pyyaml fvcore pillow==8.3.2
```

### Image Classification
- Prepare the ImageNet dataset in the timm format (DATASET_DIR/train/ DATASET_DIR/val/).

- Set the following ENV variable:
  ```
  $MASTER_ADDR: IP address of the node 0 (Not required if you have only one node (machine))
  $MASTER_PORT: Port used for initializing distributed environment
  $NODE_RANK: Index of the node
  $N_NODES: Number of nodes 
  $NPROC_PER_NODE: Number of GPUs (NOTE: should exactly match local GPU numbers with `CUDA_VISIBLE_DEVICES`)
  ```

- Training:
  - Example1 (One machine with 8 GPUs):
  ```shell
  python -u -m torch.distributed.launch --nproc_per_node=8 \
  --nnodes=1 --node_rank=0 --master_port=12345 \
  train.py DATASET_DIR --model DaViT_tiny --batch-size 128 --lr 1e-3 \
  --native-amp --clip-grad 1.0 --output OUTPUT_DIR
  ```

  - Example2 (Two machines, each has 8 GPUs):
  ```shell
  # Node 1: (IP: 192.168.1.1, and has a free port: 12345)
  python -u -m torch.distributed.launch --nproc_per_node=8
  --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
  --master_port=12345 train.py DATASET_DIR --model DaViT_tiny --batch-size 128 --lr 2e-3 \
  --native-amp --clip-grad 1.0 --output OUTPUT_DIR

  # Node 2:
  python -u -m torch.distributed.launch --nproc_per_node=8
  --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
  --master_port=12345 train.py DATASET_DIR --model DaViT_tiny --batch-size 128 --lr 2e-3 \
  --native-amp --clip-grad 1.0 --output OUTPUT_DIR
  ```

- Validation:
  ```shell
  CUDA_VISIBLE_DEVICES=0 python -u validate.py DATASET_DIR --model DaViT_tiny --batch-size 128  \
  --native-amp  --checkpoint TRAINED_MODEL_PATH  # --img-size 224 --no-test-pool
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
| DaViT-T | IN-1K | 224 | 82.8 | 96.2 | 28.3M   | 4.5G   | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EcJYlVYP9y5FsDbMP-nwjtABgLmdrNGZBxgR5--Kg3MZ7Q?e=v9OrXB) | [log](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/u3007305_connect_hku_hk/ESUeO_rmjHBFtO0a6wWzCWEB95GwCCfEOVH0vLMDRAP9JA?e=TD5Ya3) |
| DaViT-S | IN-1K | 224 | 84.2 | 96.9 | 49.7M   | 8.8G   | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/ERN_Gzm8oKdIvO01r0X2cscB_owuRAnnQPNGJEVGP23xHQ?e=MVW3ZO) | [log](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/u3007305_connect_hku_hk/Ef8aYlSGR_NAk9W1sVEtDk0Btsd5-Gqqii7su-w0gcLIhg?e=2HyRkz) |
| DaViT-B | IN-1K | 224 | 84.6 | 96.9 | 87.9M   | 15.5G  | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/Ea8kCpdM949CvTM0w7DWbGwBxxiACMSik4zx-emrNY0uKQ?e=aEjC7u) | [log](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/u3007305_connect_hku_hk/EWULh3VtXs9AnsU5ffHt9CQB6wVFCljUyHrsEqdbxs08XA?e=gnlpss) |

### Object Detection and Instance Segmentation on [COCO](https://cocodataset.org/#home)

#### [Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)

| Backbone | Pretrain | Lr Schd | #params | FLOPs | box mAP | mask mAP | Checkpoint | Log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DaViT-T | ImageNet-1K | 1x | 47.8M | 263G | 45.0 | 41.1 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/ERgc6486IzJPpPWDPJUwKP0BCc4oFtTgdudnPmmPqozqug?e=QWsLYg) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/ESQazYQeawxEgnbltKaHatYBx3D3n_LoY5BbAJFZ7z3B9w?e=kLCuGA) |
| DaViT-T | ImageNet-1K | 3x | 47.8M | 263G | 47.4 | 42.9 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/Ed1Su_p29_5KuX5Lbc2oq6gB3AVKnI-ojAT-yyICXWRbgg?e=XdJPsp) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EWErftNUOWhLmbgupYJqpN4BUwsbGHyGw5G7W4z0ofcszQ?e=Y7kY8Q) |
| DaViT-S | ImageNet-1K | 1x | 69.2M | 351G | 47.7 | 42.9 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EVgVH-9l--pJpBuly1xwsAkBx57Ph9ZAajt31vk-TqQPrA?e=byv2uC) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/ES1K98hjxVlPgbGb0ltuNrkB-mOt12p7fqeCavKBZCQKCw?e=JT9Yop) |
| DaViT-S | ImageNet-1K | 3x | 69.2M | 351G | 49.5 | 44.3 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EQCvUkIhwxNOrR1KibewNU0BqDDxzt311xANejUHprm0yQ?e=LiK9hX) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EZSLgn5rhLVFpytiMPhSJOYBWG9Ce8_OEXopJ-la7IwdGA?e=rJ4CNO) |
| DaViT-B | ImageNet-1K | 1x | 107.3M | 491G | 48.2 | 43.3 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EY5MU9O8n0ZPrvzCgakS4PoBdg6H3FCOSi1QVNCpJSQJ7g?e=MtdUGz) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EdzPfkDF-vVEufD1o-9FW8QBFba2wyZZ48zeLP6UWemPpQ?e=Gff1dt) |
| DaViT-B | ImageNet-1K | 3x | 107.3M | 491G | 49.9 | 44.6 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/Ed8kh4c1G41OmfIzo5sOS74Bccp1G65qg0H2s0WtcXUJZA?e=lnqLoQ) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EWWLBQa-_rFAuT0GzkxdP28B7_fP0vZvY-M2koNBtvijHg?e=4t7Ts3) |

#### [RetinaNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

| Backbone | Pretrain | Lr Schd | #params | FLOPs | box mAP | Checkpoint | Log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DaViT-T | ImageNet-1K | 1x | 38.5M | 244G | 44.0 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EdNCWmdUArpPgjNn2JY37_QBs5KVNy8d-8bNK7DNyjYHyg?e=osMprJ) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/Eb-vKCFfBqFPhR8ttWhPwCQBcvOzTLMR75vhdEPr_FzpYQ?e=97f2B5) |
| DaViT-T | ImageNet-1K | 3x | 38.5M | 244G | 46.5 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EZ10htYokJhOi0serTLAx5cBGcJLL7PgWSO1uzlIXUw1Jg?e=zsMh4y) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EX_D7yv9GNVEhFhlk0uWprsBbC1IkMlWqBMDDvEryWwB2Q?e=Q0zzsm) |
| DaViT-S | ImageNet-1K | 1x | 59.9M | 332G | 46.0 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EfWFiJY0iEpGhPzwQA5nv7IB-8bgnVfCHQVT4FqX8UeK9w?e=sr0Ed6) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EYwX4Uh0_SRJq7UPNqOTWHoBC5wyELg5tZOWZ9MX8QL8tg?e=wKWfpW) |
| DaViT-S | ImageNet-1K | 3x | 59.9M | 332G | 48.2 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/Ef9u9l_81_1PvCKGngnrkPoB1mGjelED0CzXr7RJwoQB7w?e=zTYozh) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/ETHo6vg3H1dOvk6Gr_9pAZMBYklfAQ4X5KmlmjEDxcii9A?e=47QUWV) |
| DaViT-B | ImageNet-1K | 1x | 98.5M | 471G | 46.7 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/ESKE9OFCDzxGo2D2Eyr29c0BFhc-j8OjF2WRoSxSK0gn4g?e=KfyJCA) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EVIF2HSXTDlJuJ1CBgwtrsEBCJdkyjVYJDcdlfr6I25kaQ?e=391Nf8) |
| DaViT-B | ImageNet-1K | 3x | 98.5M | 471G | 48.7 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/Ea7c6g_OkqpHrGIviK2Ua-4B2wc4Rd0PTdXAlA2IyTZxGQ?e=N4askR) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/ES1q-avbGmBKvqy1qM5M-ukB8U8vM4lJCuJDKy5T_FmtcA?e=CqhveA) |

### Semantic Segmentation on [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

| Backbone | Pretrain  | Method | Resolution | Iters | #params | FLOPs | mIoU | Checkpoint | Log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DaViT-T | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 60M  | 940G | 46.3 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/ERajqEKtzIBGgX2slZxCYrEBoEL-ZnLkUaXP9SHT-rp13w?e=Ga9U8a) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/Ecg2QJ_3tvxOiv62IFnIZ-sBoQp-VaEWCgJ590K9QxC6Lw?e=nD1i21) |
| DaViT-S | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 81M | 1030G | 48.8 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/Ef5hJPsJXNZLiRkUTL9-ozcBaVyVEAvolNwMUlsmJhm2Yg?e=BnJ6Et) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EfUvOZTjLA5PriA7dtLIt-8BoWgqUx8nykqwYr0GduY9Cg?e=XbSPQZ) |
| DaViT-B | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 121M | 1175G | 49.4 | [download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EWl80Je0fUlNlbJzI_qKgIkBFX-epbfK3Vzgdq1C2iOsuA?e=wumOBS) | [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3007305_connect_hku_hk/EdOwVqcgjcZOs-HHTcPqcCYBXh2zr5lwsmiXbnqaBa5mbQ?e=0esjA2) |


## Citation

If you find this repo useful to your project, please consider citing it with the following bib:

    @inproceedings{ding2022davit,
      title={Davit: Dual attention vision transformers},
      author={Ding, Mingyu and Xiao, Bin and Codella, Noel and Luo, Ping and Wang, Jingdong and Yuan, Lu},
      booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXIV},
      pages={74--92},
      year={2022},
      organization={Springer}
    }

## Acknowledgement

Our codebase is built based on [timm](https://github.com/rwightman/pytorch-image-models), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We thank the authors for the nicely organized code!
