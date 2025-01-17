# <center>Equipping Graph Autoencoders: Revisiting Masking Strategies from a Robustness Perspective</center>
***This work has been accepted by SDM 2025 (SIAM International Conference on Data Mining)*** This repository contains the code for our paper ***"Equipping Graph Autoencoders: Revisiting Masking Strategies from a Robustness Perspective"***

![FilterMGAE](https://github.com/GTLSysGraph/FilterMGAE/raw/master/FilterMGAE_Framework.png)



## 🌵 Dependencies
We recommend running on Linux systems (e.g. Ubuntu and CentOS). Other systems (e.g., Windows and macOS) have not been tested.

### 🔧 **Pytorch**
**Setup Python Environment** This work is built based on PyTorch. You can install PyTorch following the instruction in PyTorch. For example:
```
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```

### 🔧 **DGL**
DGL is an easy-to-use, high performance and scalable Python package for deep learning on graphs.

you can install DGL via:
```
conda install -c dglteam dgl-cuda10.2==0.6.1
```

### 🔧 **Install requirements**
After setting up the environment as described above, you can directly install the required packages using the `requirements.txt` file.
```
pip install -r requirements.txt
```

## 🚀 Train & Test
We provide the Cora dataset attacked by MetaAttack. Download the dataset from [Google Drive](https://drive.google.com/drive/my-drive), and change the data dir in file `dataset_dgl/datasets_file/cora.py` to the directory where your files are located.

Then, we provide the script for pretraining and performing robustness testing using the trained encoder.
```
sh train.sh
```

## 🚟 Acknowledgments
During our implementations, we referred the following code and we sincerely appreciate their valuable contributions:

https://github.com/THUDM/GraphMAE \
https://github.com/THUDM/GraphMAE2

## 🚁 Citation 
If you find this work helpful in your research, please consider citing our paper! : )


