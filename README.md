# EMS-GCN-hyperspectral-image-classification
Demo code of "EMS-GCN: An End-to-End Mixhop Superpixel-Based Graph Convolutional Network for Hyperspectral Image Classification"

## Step 1: compiling cuda files
```
cd lib
. install.sh ## please wait for about 5 minutes
```
you can also refer to [ESCNet](https://github.com/Bobholamovic/ESCNet) for the compiling process.

## Step2: train and test
```
cd ..
CUDA_VISIBLE_DEVICES='7' python main.py
```

## Step3: record classification result
![image](https://github.com/immortal13/EMS-GCN-hyperspectral-image-classification/assets/44193495/da4a1091-3180-4fc3-b4dc-7926b2835819)

## Citation
If you find this work interesting in your research, please kindly cite:
```
@ARTICLE{9745164,  
  author={Zhang, Hongyan and Zou, Jiaqi and Zhang, Liangpei},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},   
  title={EMS-GCN: An End-to-End Mixhop Superpixel-Based Graph Convolutional Network for Hyperspectral Image Classification},   
  year={2022},  
  volume={60},  
  number={},  
  pages={1-16},  
  doi={10.1109/TGRS.2022.3163326}}
```
Thank you very much! (*^▽^*)

Or if you have any questions, please feel free to contact me (Jiaqi Zou, immortal@whu.edu.cn).

