# Car Brand Classification

This is a PyTorch implementation for the homwwork of a class: "Selected Topics in Visual Recognition using Deep Learning".
I refer the ECCV2018 paper "Learning to Navigate for Fine-grained Classification" and its implementation.
Here is a citation:
<pre><code>@inproceedings{Yang2018Learning,
author = {Yang, Ze and Luo, Tiange and Wang, Dong and Hu, Zhiqiang and Gao, Jun and Wang, Liwei},
title = {Learning to Navigate for Fine-grained Classification},
booktitle = {ECCV},
year = {2018}
}
</code></pre>

## Kaggle competition link
https://www.kaggle.com/t/14e99b9514d74996b6b04df4fed0ed19

## Installation
<pre><code>conda install -c conda-forge jupyterlab
conda install -c anaconda git
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c menpo imageio
conda install -c anaconda pandas
conda install -c bioconda p7zip
conda install -c conda-forge kaggle

git clone https://github.com/chilung/NCTU_Adv_DNN_HW1.git
</code></pre>

## Datasets
Download the dataset from kaggle using the command:
<pre><code>kaggle competitions download -c cs-t0828-2020-hw1
</code></pre>

unzip the dataset and put in the root directory of the project:
<pre><code>7za x cs-t0828-2020-hw1.zip
</code></pre>

## Test the model
If you want to test the model, just run ``python test.py``. You need to specify the ``test_model`` in ``config.py`` to choose the checkpoint model for testing.

## Train the model
If you want to train the model, just run ``python train.py``. You may need to change the configurations in ``config.py``. The parameter ``PROPOSAL_NUM`` is ``M`` in the original paper and the parameter ``CAT_NUM`` is ``K`` in the original paper. During training, the log file and checkpoint file will be saved in ``save_dir`` directory. You can change the parameter ``resume`` to choose the checkpoint model to resume.

## Model
We also provide the checkpoint model trained by ourselves, you can download it from [here](https://drive.google.com/file/d/1F-eKqPRjlya5GH2HwTlLKNSPEUaxCu9H/view?usp=sharing). If you test on our provided model, you will get a 87.6% test accuracy.
