# FlashST: A Simple and Universal Prompt-Tuning Framework for Traffic Prediction

A pytorch implementation for the paper: [FlashST: A Simple and Universal Prompt-Tuning Framework for Traffic Prediction](https://arxiv.org/abs/2405.17898)<br />  

[Zhonghang Li](https://scholar.google.com/citations?user=__9uvQkAAAAJ), [Lianghao Xia](https://akaxlh.github.io/), [Yong Xu](https://scholar.google.com/citations?user=1hx5iwEAAAAJ), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />  

**[Data Intelligence Lab](https://sites.google.com/view/chaoh/home)@[University of Hong Kong](https://www.hku.hk/)**, [South China University of Technology](https://www.scut.edu.cn/en/), PAZHOU LAB

-----------

## Introduction

<p style="text-align: justify">
In this work, we introduce a simple and universal spatio-temporal prompt-tuning framework, which addresses the significant challenge posed by distribution shift in this field. 
To achieve this objective, we present FlashST, a framework that adapts pretrained models to the specific characteristics of diverse downstream datasets, thereby improving generalization across various prediction scenarios.
We begin by utilizing a lightweight spatio-temporal prompt network for in-context learning, capturing spatio-temporal invariant knowledge and facilitating effective adaptation to diverse scenarios. Additionally, we incorporate a distribution mapping mechanism to align the data distributions of pre-training and downstream data, facilitating effective knowledge transfer in spatio-temporal forecasting. Empirical evaluations demonstrate the effectiveness of our FlashST across different spatio-temporal prediction tasks.

</p>

![The detailed framework of the proposed FlashST.](https://github.com/LZH-YS1998/GPT-ST_img/blob/main/FlashST.png)

-----------
<span id='Usage'/>



## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment'>2. Environment </a>
* <a href='#Run the codes'>3. Run the codes </a>

****


<span id='Code Structure'/>

### 1. Code Structure <a href='#all_catelogue'>[Back to Top]</a>


* **conf**: This folder includes parameter settings for FlashST (`config.conf`) as well as all other baseline models.
* **data**: The documentation encompasses all the datasets utilized in our work, alongside prefabricated files and the corresponding file generation codes necessary for certain baselines.
* **lib**: Including a series of initialization methods for data processing, as follows:
	* `data_process.py`: Load, split, generate data, normalization method, slicing, etc.
    * `logger.py`: For output printing.
	* `metrics.py`: Method for calculating evaluation indicators.
	* `predifineGraph.py`: Predefined graph generation method.
	* `TrainInits.py`: Training initialization, including settings of optimizer, device, random seed, etc.
* **model**: Includes the implementation of FlashST and all baseline models, along with the necessary code to support the framework's execution. The `args.py` script is utilized to generate the required prefabricated data and parameter configurations for different baselines. Additionally, the `SAVE` folder serves as the storage location for saving the pre-trained models.
* **SAVE**: This folder serves as the storage location for saving the trained models, including pretrain, eval and ori.


```
│  README.md
│  requirements.txt
│
├─conf
│  ├─AGCRN
│  ├─ASTGCN
│  ├─FlashST
│  │  │  config.conf
│  │  │  Params_pretrain.py
│  ├─GWN
│  ├─MSDR
│  ├─MTGNN
│  ├─PDFormer
│  ├─ST-WA
│  ├─STFGNN
│  ├─STGCN
│  ├─STSGCN
│  └─TGCN
│
├─data
│  ├─CA_District5
│  ├─chengdu_didi
│  ├─NYC_BIKE
│  ├─PEMS03
│  ├─PEMS04
│  ├─PEMS07
│  ├─PEMS07M
│  ├─PEMS08
│  ├─PDFormer
│  ├─STFGNN
│  └─STGODE
│
├─lib
│  │  data_process.py
│  │  logger.py
│  │  metrics.py
│  │  predifineGraph.py
│  │  TrainInits.py
│
├─model
│  │  FlashST.py
│  │  PromptNet.py
│  │  Run.py
│  │  Trainer.py
│  │
│  ├─AGCRN
│  ├─ASTGCN
│  ├─DMSTGCN
│  ├─GWN
│  ├─MSDR
│  ├─MTGNN
│  ├─PDFormer
│  ├─STFGNN
│  ├─STGCN
│  ├─STGODE
│  ├─STSGCN
│  ├─ST_WA
│  └─TGCN
│
└─SAVE
    └─pretrain
        ├─GWN
        │      GWN_P8437.pth
        │
        ├─MTGNN
        │      MTGNN_P8437.pth
        │
        ├─PDFormer
        │      PDFormer_P8437.pth
        │
        └─STGCN
                STGCN_P8437.pth
            
```

---------

<span id='Environment'/>

### 2.Environment <a href='#all_catelogue'>[Back to Top]</a>
The code can be run in the following environments, other version of required packages may also work.
* python==3.9.12
* numpy==1.23.1
* pytorch==1.9.0
* cudatoolkit==11.1.1  

Or you can install the required environment, which can be done by running the following commands:
```
# cteate new environmrnt
conda create -n FlashST python=3.9.12

# activate environmrnt
conda activate FlashST

# Torch with CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install required libraries
pip install -r requirements.txt
```

---------

<span id='Run the codes'/>

### 3. Run the codes <a href='#all_catelogue'>[Back to Top]</a>

* First, download "data" folder from [hugging face (data.zip)](https://huggingface.co/datasets/bjdwh/FlashST-DATA/tree/main) or [wisemodel (data.zip)](https://wisemodel.cn/datasets/BJDWH/FlashST-data/file), put it in the FlashST-main directory, unzip and then enter "model" folder:
```
cd model
```
* To test different models in various modes, you can execute the Run.py code. There are some examples:
```
# Evaluate the performance of MTGNN enhanced by FlashST on the PEMS07M dataset
python Run.py -dataset_test PEMS07M -mode eval -model MTGNN

# Evaluate the performance of STGCN enhanced by FlashST on the CA_District5 dataset
python Run.py -dataset_test CA_District5 -mode eval -model STGCN

# Evaluate the original performance of STGCN on the chengdu_didi dataset
python Run.py -dataset_test chengdu_didi -mode ori -model STGCN

# Pretrain from scratch with MTGNN model, checkpoint will be saved in FlashST-main/SAVE/pretrain/MTGNN(model name)/xxx.pth
python Run.py -mode pretrain -model MTGNN
```

* Parameter setting instructions. The parameter settings consist of two parts: the pre-training model and the baseline model. To avoid any confusion arising from potential overlapping parameter names, we employ a hyphen (-) to specify the parameters of FlashST and use a double hyphen (--) to specify the parameters of the baseline model. Here is an example:
```
# Set first_layer_embedding_size and out_layer_dim to 32 in STFGNN
python Run.py -model STFGNN -mode ori -dataset_test PEMS08 --first_layer_embedding_size 32 --out_layer_dim 32
```

---------


## Citation

If you find UrbanGPT useful in your research or applications, please kindly cite:

```
@misc{li2024flashst,
      title={FlashST: A Simple and Universal Prompt-Tuning Framework for Traffic Prediction}, 
      author={Zhonghang Li and Lianghao Xia and Yong Xu and Chao Huang},
      year={2024},
      eprint={2405.17898},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
---------


## Acknowledgements
We developed our code framework drawing inspiration from [AGCRN](https://github.com/LeiBAI/AGCRN) and [GPT-ST](https://github.com/HKUDS/GPT-ST). Furthermore, the implementation of the baselines primarily relies on a combination of the code released by the original author and the code from [LibCity](https://github.com/LibCity/Bigscity-LibCity). We extend our heartfelt gratitude for their remarkable contribution.
