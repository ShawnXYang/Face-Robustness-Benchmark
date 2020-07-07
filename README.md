## Face Robustness Benchmark

This repository provides a robustness evaluation on *Face Recognition* by using various adversarial attacks. These evaluations are conducted under diverse adversarial settings, incuding doding and impersonation attacks, <img src="http://latex.codecogs.com/gif.latex?\ell_{2}" /> and <img src="http://latex.codecogs.com/gif.latex?\ell_{\infty}" /> attacks, white-box and black-box attacks. More details and some findings can be reffered to our paper [DELVING INTO THE ADVERSARIAL ROBUSTNESS ON FACE RECOGNITION](https://arxiv).

## Introduction

* This repository studies various backbones (*e.g.*, [ResNet](https://arxiv.org/pdf/1512.03385.pdf), [IR](https://arxiv.org/pdf/1512.03385.pdf), [MobileNet](https://arxiv.org/pdf/1704.04861.pdf), [ShuffleNet](https://arxiv.org/pdf/1707.01083.pdf), *etc.*) and various losses (*e.g.*, Softmax,  [SphereFace](https://arxiv.org/pdf/1704.08063.pdf), [CosFace](https://arxiv.org/pdf/1801.09414.pdf), [ArcFace](https://arxiv.org/pdf/1801.07698.pdf), *etc.*). Some trained models and source codes are provided.
* This repository introduces various white-box attacks including FGSM, BIM, MIM, CW, CIM and LGC, and black-box attack methods including Evolutionary, *etc.* The attack scripts are in `benchmark/`. 
* This repository aims to help researchers understand the adversarial robustness and provide a reliable evaluate criteria for robustness of the future works on face recognition.
* [Our paper](arxiv) also provides some valuable insights for the design of more robust models in facial tasks, as well as in other metric learning tasks such as image retrieval, person re-identification, *etc*.


### Installation

* Python 3

* pip install -r requirements.txt

### Data Preparation

#### LFW

Put LFW dataset and `pairs.txt` to `data`.

```
data
|---lfw
|     |
|     |---AJ_Cook
|     |     |
|     |     |---AJ_Cook_0001.jpg
|     |
|     |---xxxx
|     |    |
...........
|---pairs.txt

```

The `pairs.txt` can be seen in [here](https://github.com/davidsandberg/facenet/blob/master/data/pairs.txt).

Then you can execute `scripts/align_image_lfw.py` to build aligned versions of LFW dataset(multiple resolutions).

```
data
|---lfw
|---lfw-112x112
|---lfw-160x160
|---lfw-112x96
|---pairs.txt
```

#### YTF

Similarily, the file structure will be as follows:

```
data
|---splits.txt
|---ytf-112x112
|---ytf-160x160
|---ytf-112x96
```

#### CFP

```
data
|---cfp-112x112
|---cfp-160x160
|---cfp-112x96
```

### White-Box Evaluation
Some examples are in `run_white.sh`, such as:

```
python benchmark/lfw/FGSM_white.py --distance=l2 --goal=dodging --model=MobileFace --eps=16 --log=log-lfw-FGSM-l2-dodging-MobileFace-white.txt 
```

Then the attack results are saved in `--log`. 

```
adv_img,tar_img,score,dist,success
1.npy,data/lfw-112x112/Abel_Pacheco/Abel_Pacheco_0004.jpg,0.21092090010643005,1.0467989629677874,1
2.npy,data/lfw-112x112/Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg,0.21074934303760529,4.202811928700617,1
3.npy,data/lfw-112x112/Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg,0.21039743721485138,2.1047161963395666,1
4.npy,data/lfw-112x112/Amber_Tamblyn/Amber_Tamblyn_0002.jpg,0.20931993424892426,1.2771732226518993,1
....
```

`score` indicates the similarity predicted by victim model,  `dist` means <img src="http://latex.codecogs.com/gif.latex?\ell_{2}" /> or <img src="http://latex.codecogs.com/gif.latex?\ell_{\infty}" /> distance, and `success` means whether this attack is successful.

#### White-Box Results

- The attack success rate vs. perturbation budget curves of the models against *dodging* attacks under the <img src="http://latex.codecogs.com/gif.latex?\ell_{2}" /> norm.

<p align="center">
  <img src="imgs/l2_dodging_acc_pert_lfw.png" alt="bounding box" width="640px">
</p>

- The attack success rate vs. perturbation budget curves of the models against impersonation attacks under the <img src="http://latex.codecogs.com/gif.latex?\ell_{2}" /> norm.

<p align="center">
  <img src="imgs/l2_impersonate_acc_pert_lfw.png" alt="bounding box" width="640px">
</p>


### Black-Box Evaluation

Some examples are in `run_black.sh`

```
# generate adversarial examples
python benchmark/lfw/FGSM_black.py --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=outputs/lfw-FGSM-l2-dodging-MobileFace --batch_size=20
# generate log file
python benchmark/lfw/run_test.py --model=Mobilenet --distance=l2 --anno=outputs/lfw-FGSM-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-FGSM-l2-dodging-MobileFace-black.txt --goal=dodging 

```

After executing the first script, the adversarial examples are saved as png files in `--output`. An annotation file (`annotation.txt`) is also saved in `--output`.

Then `run_test.py` will generate the evaluation log file in `--log`, and the format of the log file is same as log file of **White-Box Evaluation**.

#### Black-Box Results

- The attack success rates of the models against black-box dodging attacks under the <img src="http://latex.codecogs.com/gif.latex?\ell_{\infty}" /> norm.

<p align="center">
  <img src="imgs/heatmap_linf_dodging_lfw.png" alt="bounding box" width="640px">
</p>

- The attack success rates of the models against black-box impersonation attacks under the the <img src="http://latex.codecogs.com/gif.latex?\ell_{\infty}" /> norm.

<p align="center">
  <img src="imgs/heatmap_linf_impersonate_lfw.png" alt="bounding box" width="640px">
</p>


## Acknowledge
For the training procedure of Face Recognition, we mainly refer to the public code from [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch).


## Citation

    @article{xiao2020delving,
        title   =   {Delving into the Adversarial Robustness on Face Recognition},
        author  =   {Xiao},
        journal =   {arXiv preprint arXiv:2007.XXXXX},
        year    =   {2020}
    }
