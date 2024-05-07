# Vision Mamba: A Comprehensive Survey and Taxonomy



> **Abstract:** *State Space Model (SSM) is a mathematical model used to describe and analyze the behavior of dynamic systems. This model has witnessed numerous applications in several fields, including control theory, signal processing, economics and machine learning. In the field of deep learning, state space models are used to process sequence data, such as time series analysis, natural language processing (NLP) and video understanding. By mapping sequence data to state space, long-term dependencies in the data can be better captured. In particular,  modern SSMs have shown strong representational capabilities in NLP, especially in long sequence modeling, while maintaining linear time complexity. Notably, based on the latest state-space models, Mamba \cite{Mamba} merges time-varying parameters into SSMs and formulates a hardware-aware algorithm for efficient training and inference. Given its impressive efficiency and strong long-range dependency modeling capability, Mamba is expected to become a new AI architecture that may outperform Transformer. Recently, a number of works have attempted to study the potential of Mamba in various fields, such as general vision, multi-modal, medical image analysis and remote sensing image analysis, by extending Mamba from natural language domain to visual domain. To fully understand Mamba in the visual domain, we conduct a comprehensive survey and present a taxonomy study. This survey focuses on Mamba's application to a variety of visual tasks and data types, and discusses its predecessors, recent advances and far-reaching impact on a wide range of domains. Since Mamba is now on an upward trend, please actively notice us if you have new findings, and new progress on Mamba will be included in this survey in a timely manner and updated on the website: (https://github.com/lx6c78/Vision-Mamba-A-Comprehensive-Survey-and-Taxonomy).*



:star: **We will timely update the latest representaive literatures and their released source code on this page. If you have any questions, please don't hesitate to contact us at any of the following emails:**



## 📢 Update Log

- 



## Citation

If you find this repository is useful for you, please cite our paper:

```

```



## Contents

- [Related Survey](#Related-Survey)
  - [High-level/Mid-level Vision](#1-High-levelMid-level-Vision)
    - [Vision Backbone with Mamba](#11-Vision-Backbone-with-Mamba)
    - [Video Analysis and Understanding](#12-Video-Analysis-and-Understanding)
    - [Down-stream Visual Applications](#13-Down-stream-Visual-Applications)
  - [Low-level Vision](#2-Low-level-Vision)
    - [Image Denoising](#21-Image-Denoising)
    - [Image Restoration](#22-Image-Restoration)
  - [3-D Visual Recognition](#3-3-D-Visual-Recognition)
    - [Point Could Analysis](#31-Point-Could-Analysis)
    - [Hyperspectral Imaging Analysis](#32-Hyperspectral-Imaging-Analysis)
  - [Visual Data Generation](#4-Visual-Data-Generation)
- [Multi-Modal](#Multi-Modal)
  - [Heterologous Stream](#1-Heterologous-Stream)
    - [Multi-Modal Understanding](#11-Multi-Modal-Understanding)
    - [Multimodal large language models](#12-Multimodal-large-language-models)
  - [Homologous Stream](#2-Homologous-Stream)
- [Vertical Application](#Vertical-Application)
  - [Remote Sensing Image](#1-Remote-Sensing-Image)
    - [Remote Sensing Image Processing](#11-Remote-Sensing-Image-Processing)
    - [Remote Sensing Image Classification](#12-Remote-Sensing-Image-Classification)
    - [Remote Sensing Image Change Detection](#13-Remote-Sensing-Image-Change-Detection)
    - [Remote Sensing Image Segmentation](#14-Remote-Sensing-Image-Segmentation)
    - [Remote Sensing Image Fusion](#15-Remote-Sensing-Image-Fusion)
  - [Medical Image](#2-Medical-Image)
    - [Medical Image Segmentation](#21-Medical-Image-Segmentation)
      - [Preliminary explorations of U-shaped Mamba](#211-Preliminary-explorations-of-U-shaped-Mamba)
      - [Improvements to the U-shaped Mamba](#212-Improvements-to-the-U-shaped-Mamba)
      - [U-shaped Mamba with other methodologies](#213-U-shaped-Mamba-with-other-methodologies)
      - [Multi-Dimensional Medical Data Segmentation](#214-Multi-Dimensional-Medical-Data-Segmentation)
    - [Pathological Diagnosis](#22-Pathological-Diagnosis)
    - [Deformable Image Registration](#23-Deformable-Image-Registration)
    - [Medical Image Reconstruction](#24-Medical-Image-Reconstruction)
    - [Other Medical Tasks](#25-Other-Medical-Tasks)
- [Other Domains](#other-domains) <br/>




## Related Survey

- **State Space Model for New-Generation Network Alternative to Transformers: A Survey.** [15 April 2024] [ArXiv, 2024]<br/>
  *Xiao Wang, Shiao Wang, Yuhe Ding, Yuehang Li, Wentao Wu, Yao Rong,  Weizhe Kong, Ju Huang, Shihao Li, Haoxiang Yang, Ziwen Wang, Bowei Jiang,  Chenglong Li, Yaowei Wang, Yonghong Tian, Jin Tang.*<br/>[[Paper](https://arxiv.org/abs/2404.09516)] [[Github](https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List)]
- **A Survey on Visual Mamba.** [26 April, 2024] [ArXiv, 2024]<br/>
  *Hanwei Zhang, Ying Zhu, Dan Wang, Lijun Zhang, Tianxiang Chen, Zi Ye.*<br/>
  [[Paper](https://arxiv.org/abs/2404.15956)]
- **Mamba-360: Survey of State Space Models as Transformer Alternative for Long Sequence Modelling: Methods, Applications, and Challenges.** [24 April, 2024] [ArXiv, 2024]<br/>
  *Badri Narayana Patro, Vijay Srinivas Agneeswaran.*<br/>
  [[Paper](https://arxiv.org/abs/2404.16112)] [[Gihub](https://github.com/badripatro/mamba360)]
- **A Survey on Vision Mamba: Models, Applications and Challenges.** [29 April, 2024] [ArXiv, 2024]<br/>
  *Rui Xu, Shu Yang, Yihui Wang, Bo Du, Hao Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2404.18861)] [[Gihub](https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List)]




## General Vision

### 1 High-level/Mid-level Vision

#### 1.1 Vision Backbone with Mamba

- **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model.** [10 February, 2024] [ArXiv, 2024]<br/>
  *Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2401.09417)] [[Code](https://github.com/hustvl/Vim)]
- **VMamba: Visual State Space Model.** [10 April, 2024] [ArXiv, 2024]<br/>
  *Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Yunfan Liu.*<br/>
  [[Paper](https://arxiv.org/abs/2401.10166)] [[Code](https://github.com/MzeroMiko/VMamba)]
- **Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data.** [19 March, 2024] [ArXiv, 2024]<br/>
  *Shufan Li, Harkanwar Singh, Aditya Grover.*<br/>
  [[Paper](https://arxiv.org/abs/2402.05892)] [[Code](https://github.com/jacklishufan/Mamba-ND)]
- **LocalMamba: Visual State Space Model with Windowed Selective Scan.** [14 March, 2024] [ArXiv, 2024]<br/>
  *Tao Huang, Xiaohuan Pei, Shan You, Fei Wang, Chen Qian, Chang Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2403.09338)] [[Code](https://github.com/hunto/LocalMamba)]
- **EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba.** [14 March, 2024] [ArXiv, 2024]<br/>
  *Xiaohuan Pei, Tao Huang, Chang Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2403.09977)] [[Code](https://github.com/TerryPei/EfficientVMamba)]
- **SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series.** [24 April, 2024] [ArXiv, 2024]<br/>
  *Badri N. Patro, Vijay S. Agneeswaran.*<br/>
  [[Paper](https://arxiv.org/abs/2403.15360)] [[Code](https://github.com/badripatro/Simba)]
- **PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition.** [26 March, 2024] [ArXiv, 2024]<br/>
  *Chenhongyi Yang, Zehui Chen, Miguel Espinosa, Linus Ericsson, Zhenyu Wang, Jiaming Liu, Elliot J. Crowley.*<br/>
  [[Paper](https://arxiv.org/abs/2403.17695)] [[Code](https://github.com/ChenhongyiYang/PlainMamba)]
- **On the low-shot transferability of [V]-Mamba.** [15 March, 2024] [ArXiv, 2024]<br/>
  *Diganta Misra, Jay Gala, Antonio Orvieto.*<br/>
  [[Paper](https://arxiv.org/abs/2403.10696)]
- **DGMamba: Domain Generalization via Generalized State Space Model.** [11 April, 2024] [ArXiv, 2024] <br/>
  *Shaocong Long, Qianyu Zhou, Xiangtai Li, Xuequan Lu, Chenhao Ying, Yuan Luo, Lizhuang Ma, Shuicheng Yan.*<br/>
  [[Paper](https://arxiv.org/abs/2404.07794)] [[Code](https://github.com/longshaocong/DGMamba)]

#### 1.2 Video Analysis and Understanding

- **VideoMamba: State Space Model for Efficient Video Understanding.** [March, 2024] [ArXiv, 2024]<br/>
  *Kunchang Li, Xinhao Li, Yi Wang, Yinan He, Yali Wang, Limin Wang, Yu Qiao.*<br/>
  [[Paper](https://arxiv.org/abs/2403.06977)] [[Code](https://github.com/OpenGVLab/VideoMamba)]
- **Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding.** [14 March, 2024] [ArXiv, 2024]<br/>
  *Guo Chen, Yifei Huang, Jilan Xu, Baoqi Pei, Zhe Chen, Zhiqi Li, Jiahao Wang, Kunchang Li, Tong Lu, Limin Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.09626)] [[Code](https://github.com/OpenGVLab/video-mamba-suite)]
- **RhythmMamba: Fast Remote Physiological Measurement with Arbitrary Length Videos.** [9 April, 2024] [ArXiv, 2024] <br/>
  *Bochao Zou, Zizheng Guo, Xiaocheng Hu, Huimin Ma.*<br/>
  [[Paper](https://arxiv.org/abs/2404.06483)] [[Code](https://github.com/zizheng-guo/RhythmMamba)]

#### 1.3 Down-stream Visual Applications

- **Res-VMamba: Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning.** [28 April, 2024] [ArXiv, 2024] <br/>
  *Chi-Sheng Chen, Guan-Ying Chen, Dong Zhou, Di Jiang, Dai-Shi Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2402.15761)] [[Code](https://github.com/ChiShengChen/ResVMamba)]
- **InsectMamba: Insect Pest Classification with State Space Model.** [4 April, 2024] [ArXiv, 2024] <br/>
  *Qianning Wang, Chenglin Wang, Zhixin Lai, Yucheng Zhou.*<br/>[[Paper](https://arxiv.org/abs/2404.03611)]
- **MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection.** [17 March, 2024] [ArXiv, 2024]<br/>
  *Tianxiang Chen, Zhentao Tan, Tao Gong, Qi Chu, Yue Wu, Bin Liu, Jieping Ye, Nenghai Yu.*<br/>[[Paper](https://arxiv.org/abs/2403.02148)] [[Code](https://github.com/txchen-USTC/MiM-ISTD)]

### 2 Low-level Vision

#### 2.1 Image Denoising

- **U-shaped Vision Mamba for Single Image Dehazing.** [15 February, 2024] [ArXiv, 2024]<br/>
  *Zhuoran Zheng, Chen Wu.*<br/>[[Paper](https://arxiv.org/abs/2402.04139)] [[Code](https://github.com/zzr-idam)]
- **FreqMamba: Viewing Mamba from a Frequency Perspective for Image Deraining.** [15 April, 2024] [ArXiv, 2024]<br/>
  *Zou Zhen, Yu Hu, Zhao Feng.*<br/>[[Paper](https://arxiv.org/abs/2404.09476)]

#### 2.2 Image Restoration

- **MambaIR: A Simple Baseline for Image Restoration with State-Space Model.** [25 March, 2024] [ArXiv, 2024] <br/>
  *Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, Shu-Tao Xia.*<br/>
  [[Paper](https://arxiv.org/abs/2402.15648)] [[Code](https://github.com/csguoh/MambaIR)]
- **Activating Wider Areas in Image Super-Resolution.** [13 March, 2024] [ArXiv, 2024]<br/>
  *Cheng Cheng, Hang Wang, Hongbin Sun.*<br/>
  [[Paper](https://arxiv.org/abs/2403.08330)]
- **CU-Mamba: Selective State Space Models with Channel Learning for Image Restoration.** [17 April, 2024] [ArXiv, 2024]<br/>
  *Rui Deng, Tianpei Gu.*<br/>[[Paper](https://arxiv.org/abs/2404.11778)]
- **VmambaIR: Visual State Space Model for Image Restoration.** [17 March, 2024] [ArXiv, 2024]<br/>
  *Yuan Shi, Bin Xia, Xiaoyu Jin, Xing Wang, Tianyu Zhao, Xin Xia, Xuefeng Xiao, Wenming Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.11423)] [[Code](https://github.com/AlphacatPlus/VmambaIR)]

### 3 3-D Visual Recognition

#### 3.1 Point Could Analysis

- **PointMamba: A Simple State Space Model for Point Cloud Analysis.** [2 April, 2024] [ArXiv, 2024]<br/>
  *Dingkang Liang, Xin Zhou, Xinyu Wang, Xingkui Zhu, Wei Xu, Zhikang Zou, Xiaoqing Ye, Xiang Bai.*<br/>
  [[Paper](https://arxiv.org/abs/2402.10739)] [[Code](https://github.com/LMD0311/PointMamba)]
- **Point Cloud Mamba: Point Cloud Learning via State Space Model.** [29 March, 2024] [ArXiv, 2024]<br/>
  *Tao Zhang, Xiangtai Li, Haobo Yuan, Shunping Ji, Shuicheng Yan.*<br/>
  [[Paper](https://arxiv.org/abs/2403.00762)] [[Code](https://github.com/SkyworkAI/PointCloudMamba)]
- **Point Mamba: A Novel Point Cloud Backbone Based on State Space Model with Octree-Based Ordering Strategy.** [17 March, 2024] [ArXiv, 2024]<br/>
  *Jiuming Liu, Ruiji Yu, Yian Wang, Yu Zheng, Tianchen Deng, Weicai Ye, Hesheng Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.06467)] [[Code](https://github.com/IRMVLab/Point-Mamba)]
- **3DMambaComplete: Exploring Structured State Space Model for Point Cloud Completion.** [10 April, 2024] [ArXiv, 2024]<br/>
  *Yixuan Li, Weidong Yang, Ben Fei.*<br/>
  [[Paper](https://arxiv.org/abs/2404.07106)]

#### 3.2 Hyperspectral Imaging Analysis

- **Mamba-FETrack: Frame-Event Tracking via State Space Model.** [28 April, 2024] [ArXiv, 2024]<br/>
  *Ju Huang, Shiao Wang, Shuai Wang, Zhe Wu, Xiao Wang, Bo Jiang.*<br/>
  [[Paper](https://arxiv.org/abs/2404.18174)] [[Code](https://github.com/Event-AHU/Mamba_FETrack)]

### 4 Visual Data Generation

- **ZigMa: A DiT-style Zigzag Mamba Diffusion Model.** [1 April, 2024] [ArXiv, 2024]<br/>
  *Vincent Tao Hu, Stefan Andreas Baumann, Ming Gui, Olga Grebenkova, Pingchuan Ma, Johannes Fischer, Björn Ommer.*<br/>
  [[Paper](https://arxiv.org/abs/2403.13802)] [[Homepage](https://taohu.me/zigma/)] [[Code](https://github.com/CompVis/zigma)]
- **Motion Mamba: Efficient and Long Sequence Motion Generation with Hierarchical and Bidirectional Selective SSM.** [19 March, 2024] [ArXiv, 2024]<br/>
  *Zeyu Zhang, Akide Liu, Ian Reid, Richard Hartley, Bohan Zhuang, Hao Tang.*<br/>[[Paper](https://arxiv.org/abs/2403.07487)] [[Homepage](https://steve-zeyu-zhang.github.io/MotionMamba/)] [[Code](https://github.com/steve-zeyu-zhang/MotionMamba/)]
- **Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction.** [29 March, 2024] [ArXiv, 2024]<br/>
  *Qiuhong Shen, Xuanyu Yi, Zike Wu, Pan Zhou, Hanwang Zhang, Shuicheng Yan, Xinchao Wang.*<br/>[[Paper](https://arxiv.org/abs/2403.18795)]
- **Matten: Video Generation with Mamba-Attention.**  [5 May, 2024] [ArXiv, 2024]<br/>
  *Yu Gao, Jiancheng Huang, Xiaopeng Sun, Zequn Jie, Yujie Zhong, Lin Ma.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03025)]




## Multi-Modal

### 1 Heterologous Stream

#### 1.1 Multi-Modal Understanding

- **MambaTalk: Efficient Holistic Gesture Synthesis with Selective State Space Models.** [14 March, 2024] [ArXiv, 2024]<br/>
  *Zunnan Xu, Yukang Lin, Haonan Han, Sicheng Yang, Ronghui Li, Yachao Zhang, Xiu Li.*<br/>
  [[Paper](https://arxiv.org/abs/2403.09471)]
- **ReMamber: Referring Image Segmentation with Mamba Twister.** [26 March, 2024] [ArXiv, 2024]<br/>
  *Yuhuan Yang, Chaofan Ma, Jiangchao Yao, Zhun Zhong, Ya Zhang, Yanfeng Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.17839)]
- **SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding.** [1 April, 2024] [ArXiv, 2024]<br/>
  *Wenrui Li, Xiaopeng Hong, Xiaopeng Fan.*<br/>
  [[Paper](https://arxiv.org/abs/2404.01174)]

#### 1.2 Multimodal large language models

- **VL-Mamba: Exploring State Space Models for Multimodal Learning.** [20 March, 2024] [ArXiv, 2024]<br/>
  *Yanyuan Qiao, Zheng Yu, Longteng Guo, Sihan Chen, Zijia Zhao, Mingzhen Sun, Qi Wu, Jing Liu.*<br/>
  [[Paper](https://arxiv.org/abs/2403.13600)] [[Homepage](https://yanyuanqiao.github.io/vl-mamba/)] [[Code](https://github.com/ZhengYu518/VL-Mamba)]
- **Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference.** [22 March, 2024] [ArXiv, 2024]<br/>
  *Han Zhao, Min Zhang, Wei Zhao, Pengxiang Ding, Siteng Huang, Donglin Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.14520)] [[Homepage](https://sites.google.com/view/cobravlm)] [[Code](https://github.com/h-zhao1997/cobra)]

### 2 Homologous Stream

- **Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation.** [5 April, 2024] [ArXiv, 2024]<br/>
  *Zifu Wan, Yuhao Wang, Silong Yong, Pingping Zhang, Simon Stepputtis, Katia Sycara, Yaqi Xie.*<br/>
  [[Paper](https://arxiv.org/abs/2404.04256)] [[Code](https://github.com/zifuwan/Sigma)]
- **Fusion-Mamba for Cross-modality Object Detection.** [14 April, 2024] [ArXiv, 2024]<br/>
  *Wenhao Dong, Haodong Zhu, Shaohui Lin, Xiaoyan Luo, Yunhang Shen, Xuhui Liu, Juan Zhang, Guodong Guo, Baochang Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2404.09146)]



## Vertical Application

### 1 Remote Sensing Image

#### 1.1 Remote Sensing Image Processing

- **Pan-Mamba: Effective pan-sharpening with State Space Model.** [8 March, 2024] [ArXiv, 2024]<br/>
  *Xuanhua He, Ke Cao, Keyu Yan, Rui Li, Chengjun Xie, Jie Zhang, Man Zhou.*<br/>
  [[Paper](https://arxiv.org/abs/2402.12192)] [[Code](https://github.com/alexhe101/Pan-Mamba)]
- **HSIDMamba: Exploring Bidirectional State-Space Models for Hyperspectral Denoising.** [15 April, 2024] [ArXiv, 2024]<br/>
  *Yang Liu, Jiahua Xiao, Yu Guo, Peilin Jiang, Haiwei Yang, Fei Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2404.09697)]

#### 1.2 Remote Sensing Image Classification

- **RSMamba: Remote Sensing Image Classification with State Space Model.** [28 March, 2024] [ArXiv, 2024]<br/>
  *Keyan Chen, Bowen Chen, Chenyang Liu, Wenyuan Li, Zhengxia Zou, Zhenwei Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2403.19654)]
- **SpectralMamba: Efficient Mamba for Hyperspectral Image Classification.** [12 April, 2024] [ArXiv, 2024]<br/>
  *Jing Yao, Danfeng Hong, Chenyu Li, Jocelyn Chanussot.*<br/>
  [[Paper](https://arxiv.org/abs/2404.08489)] [[Code](https://github.com/danfenghong/SpectralMamba)]
- **Spectral-Spatial Mamba for Hyperspectral Image Classification.** [29 Apr,  2024] [ArXiv, 2024]<br/>
  *Lingbo Huang, Yushi Chen, Xin He.*<br/>
  [[Paper](https://arxiv.org/abs/2404.18401)] [[Code](https://github.com/danfenghong/SpectralMamba)]
- **S2Mamba: A Spatial-spectral State Space Model for Hyperspectral Image Classification.** [28 April, 2024] [ArXiv, 2024]<br/>
  *Guanchun Wang, Xiangrong Zhang, Zelin Peng, Tianyang Zhang, Xiuping Jia, Licheng Jiao.*<br/>
  [[Paper](https://arxiv.org/abs/2404.18213)] [[Code](https://github.com/PURE-melo/S2Mamba)]

#### 1.3 Remote Sensing Image Change Detection

- **ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model.** [14 April, 2024] [ArXiv, 2024]<br/>
  *Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, Naoto Yokoya.*<br/>
  [[Paper](https://arxiv.org/abs/2404.03425)] [[Code](https://github.com/ChenHongruixuan/MambaCD)]
- **RSCaMa: Remote Sensing Image Change Captioning with State Space Model.** [2 May, 2024] [ArXiv, 2024]<br/>
  *Chenyang Liu, Keyan Chen, Bowen Chen, Haotian Zhang, Zhengxia Zou, Zhenwei Shi.*<br/>[[Paper](https://arxiv.org/abs/2404.18895)] [[Code](https://github.com/Chen-Yang-Liu/RSCaMa)]

#### 1.4 Remote Sensing Image Segmentation

- **Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model.** [11 April, 2024] [ArXiv, 2024]<br/>
  *Qinfeng Zhu, Yuanzhi Cai, Yuan Fang, Yihan Yang, Cheng Chen, Lei Fan, Anh Nguyen.*<br>
  [[Paper](https://arxiv.org/abs/2404.01705)] [[Code](https://github.com/zhuqinfeng1999/Samba)]
- **RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation.** [3 April, 2024] [ArXiv, 2024]<br/>
  *Xianping Ma, Xiaokang Zhang, Man-On Pun.*<br/>
  [[Paper](https://arxiv.org/abs/2404.02457)] [[Code](https://github.com/sstary/SSRS)]
- **RS-Mamba for Large Remote Sensing Image Dense Prediction.** [10 April, 2024] [ArXiv, 2024]<br/>
  *Sijie Zhao, Hao Chen, Xueliang Zhang, Pengfeng Xiao, Lei Bai, Wanli Ouyang.*<br/>[[Paper](https://arxiv.org/abs/2404.02668)] [[Code](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba)]

#### 1.5 Remote Sensing Image Fusion

- **FusionMamba: Efficient Image Fusion with State Space Model.** [11 April, 2024] [ArXiv, 2024]<br/>
  *Siran Peng, Xiangyu Zhu, Haoyu Deng, Zhen Lei, Liang-Jian Deng.*<br/>
  [[Paper](https://arxiv.org/abs/2404.07932)]
- **A Novel State Space Model with Local Enhancement and State Sharing for Image Fusion.** [14 April, 2024] [ArXiv, 2024]<br/>
  *Zihan Cao, Xiao Wu, Liang-Jian Deng, Yu Zhong.*<br/>
  [[Paper](https://arxiv.org/abs/2404.09293)]

### 2 Medical Image

#### 2.1 Medical Image Segmentation

##### 2.1.1 Preliminary explorations of U-shaped Mamba

- **U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation.** [9 January, 2024] [ArXiv, 2024]<br/>
  *Jun Ma, Feifei Li, Bo Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2401.04722)] [[Homepage](https://wanglab.ai/u-mamba.html)] [[Code](https://github.com/bowang-lab/U-Mamba)]
- **VM-UNet: Vision Mamba UNet for Medical Image Segmentation.** [4 February, 2024] [ArXiv, 2024]<br/>
  *Jiacheng Ruan, Suncheng Xiang.*<br/>
  [[Paper](https://arxiv.org/abs/2402.02491)] [[Code](https://github.com/JCruan519/VM-UNet)]
- **Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation.** [30 March, 2024] [ArXiv, 2024]<br/>
  *Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, Ge Cui, Lei Li.*<br/>
  [[Paper](https://arxiv.org/abs/2402.05079)] [[Code](https://github.com/ziyangwang007/Mamba-UNet)]
- **Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining.** [6 March, 2024] [ArXiv, 2024]<br/>
  *Jiarun Liu, Hao Yang, Hong-Yu Zhou, Yan Xi, Lequan Yu, Yizhou Yu, Yong Liang, Guangming Shi, Shaoting Zhang, Hairong Zheng, Shanshan Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2402.03302)] [[Code](https://github.com/JiarunLiu/Swin-UMamba)]

##### 2.1.2 Improvements to the U-shaped Mamba

- **LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation.** [11 March, 2024] [ArXiv, 2024]<br/>
  *Weibin Liao, Yinghao Zhu, Xinyuan Wang, Chengwei Pan, Yasha Wang, Liantao Ma.*<br/>
  [[Paper](https://arxiv.org/abs/2403.05246)] [[Code](https://github.com/MrBlankness/LightM-UNet)]
- **VM-UNET-V2 Rethinking Vision Mamba UNet for Medical Image Segmentation .** [14 March, 2024] [ArXiv, 2024]<br/>
  *Mingya Zhang, Yue Yu, Limei Gu, Tingsheng Lin, Xianping Tao.*<br/>
  [[Paper](https://arxiv.org/abs/2403.09157)] [[Code](https://github.com/nobodyplayer1/VM-UNetV2)]
- **Large Window-based Mamba UNet for Medical Image Segmentation: Beyond Convolution and Self-attention.** [12 March, 2024] [ArXiv, 2024]<br/>
  *Jinhong Wang, Jintai Chen, Danny Chen, Jian Wu.*<br/>
  [[Paper](https://arxiv.org/abs/2403.07332)] [[Code](https://github.com/wjh892521292/LMa-UNet)]
- **H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation.** [20 March, 2024] [ArXiv, 2024]<br/>
  *Renkai Wu, Yinghao Liu, Pengchen Liang, Qing Chang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.13642)] [[Code](https://github.com/wurenkai/H-vmunet)]
- **Integrating Mamba Sequence Model and Hierarchical Upsampling Network for Accurate Semantic Segmentation of Multiple Sclerosis Legion.** [26 Mar, 2024] [ArXiv, 2024]<br/>
  *Kazi Shahriar Sanjid, Md. Tanzim Hossain, Md. Shakib Shahariar Junayed, Dr. Mohammad Monir Uddin.*<br/>
  [[Paper](https://arxiv.org/abs/2403.17432)]
- **Rotate to Scan: UNet-like Mamba with Triplet SSM Module for Medical Image Segmentation.** [16 April, 2024] [ArXiv, 2024]<br/>
  *Hao Tang, Lianglun Cheng, Guoheng Huang, Zhengguang Tan, Junhao Lu, Kaihong Wu.*<br/>
  [[Paper](https://arxiv.org/abs/2403.17701)]
- **UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation.** [24 April, 2024] [ArXiv, 2024]<br/>
  *Renkai Wu, Yinghao Liu, Pengchen Liang, Qing Chang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.20035)] [[Code](https://github.com/wurenkai/UltraLight-VM-UNet)]

##### 2.1.3 U-shaped Mamba with other methodologies

- **Semi-Mamba-UNet: Pixel-Level Contrastive and Pixel-Level Cross-Supervised Visual Mamba-based UNet for Semi-Supervised Medical Image Segmentation.** [29 March, 2024] [ArXiv, 2024]<br/>
  *Chao Ma, Ziyang Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2402.07245)] [[Code](https://github.com/ziyangwang007/Mamba-UNet)]
- **Weak-Mamba-UNet: Visual Mamba Makes CNN and ViT Work Better for Scribble-based Medical Image Segmentation.** [16 February, 2024] [ArXiv, 2024]<br/>
  *Ziyang Wang, Chao Ma.*<br/>
  [[Paper](https://arxiv.org/abs/2402.10887)] [[Code](https://github.com/ziyangwang007/Mamba-UNet)]
- **ProMamba: Prompt-Mamba for polyp segmentation.** [26 March, 2024] [ArXiv, 2024]<br/>
  *Jianhao Xie, Ruofan Liao, Ziang Zhang, Sida Yi, Yuesheng Zhu, Guibo Luo.*<br/>
  [[Paper](https://arxiv.org/abs/2403.13660)]
- **P-Mamba: Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation.** [15 March, 2024] [ArXiv, 2024]<br/>
  *Zi Ye, Tianxiang Chen, Fangyijie Wang, Hanwei Zhang, Guanxi Li, Lijun Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2402.08506)]

##### 2.1.4 Multi-Dimensional Medical Data Segmentation

- **SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation.** [25 February, 2024] [ArXiv, 2024]<br/>
  *Zhaohu Xing, Tian Ye, Yijun Yang, Guang Liu, Lei Zhu.*<br/>
  [[Paper](https://arxiv.org/abs/2401.13560)] [[Code](https://github.com/ge-xing/SegMamba)]
- **nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model.** [10 March, 2024] [ArXiv, 2024]<br/>
  *Haifan Gong, Luoyao Kang, Yitao Wang, Xiang Wan, Haofeng Li.*<br/>
  [[Paper](https://arxiv.org/abs/2402.03526)] [[Code](https://github.com/lhaof/nnMamba)]
- **T-Mamba: Frequency-Enhanced Gated Long-Range Dependency for Tooth 3D CBCT Segmentation.** [1 April, 2024] [ArXiv, 2024]<br/>
  *Jing Hao, Lei He, Kuo Feng Hung.*<br/>
  [[Paper](https://arxiv.org/abs/2404.01065)] [[Code](https://github.com/isbrycee/T-Mamba)]
- **Vivim: a Video Vision Mamba for Medical Video Object Segmentation.** [12 March, 2024] [ArXiv, 2024]<br/>
  *Yijun Yang, Zhaohu Xing, Chunwang Huang, Lei Zhu.*<br/>
  [[Paper](https://arxiv.org/abs/2401.14168)] [[Code](https://github.com/scott-yjyang/Vivim)]

#### 2.2 Pathological Diagnosis

- **MedMamba: Vision Mamba for Medical Image Classification.** [2 April, 2024] [ArXiv, 2024]<br/>
  *Yubiao Yue, Zhenzhang Li.* <br/>
  [[Paper](https://arxiv.org/abs/2403.03849)] [[Code](https://github.com/YubiaoYue/MedMamba)]
- **MamMIL: Multiple Instance Learning for Whole Slide Images with State Space Models.** [8 March, 2024] [ArXiv, 2024]<br/>
  *Zijie Fang, Yifeng Wang, Zhi Wang, Jian Zhang, Xiangyang Ji, Yongbing Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.05160)]
- **MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology.** [11 March, 2024] [ArXiv, 2024]<br/>
  *Shu Yang, Yihui Wang, Hao Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2403.06800)] [[Code](https://github.com/isyangshu/MambaMIL)]
- **CMViM: Contrastive Masked Vim Autoencoder for 3D Multi-modal Representation Learning for AD classification.** [25 March, 2024] [ArXiv, 2024]<br/>
  *Guangqian Yang, Kangrui Du, Zhihan Yang, Ye Du, Yongping Zheng, Shujun Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2403.16520)]
- **SurvMamba: State Space Model with Multi-grained Multi-modal Interaction for Survival Prediction.** [11 April, 2024] [ArXiv, 2024]<br/>
  *Ying Chen, Jiajing Xie, Yuxiang Lin, Yuhang Song, Wenxian Yang, Rongshan Yu.*<br/>
  [[Paper](https://arxiv.org/abs/2404.08027)]

#### 2.3 Deformable Image Registration

- **MambaMorph: a Mamba-based Framework for Medical MR-CT Deformable Registration.** [12 March, 2024] [ArXiv, 2024]<br/>
  *Tao Guo, Yinuo Wang, Shihao Shu, Diansheng Chen, Zhouping Tang, Cai Meng, Xiangzhi Bai.*<br/>
  [[Paper](https://arxiv.org/abs/2401.13934)] [[Code](https://github.com/Guo-Stone/MambaMorph)]
- **VMambaMorph: a Visual Mamba-based Framework with Cross-Scan Module for Deformable 3D Image Registration.** [7 Apr, 2024] [ArXiv, 2024]<br/>
  *Ziyang Wang, Jian-Qing Zheng, Chao Ma, Tao Guo.*<br/>
  [[Paper](https://arxiv.org/abs/2404.05105v2)] [[Code](https://github.com/ziyangwang007/VMambaMorph)]

#### 2.4 Medical Image Reconstruction

- **FD-Vision Mamba for Endoscopic Exposure Correction.** [14 February, 2024] [ArXiv, 2024]<br/>
  *Zhuoran Zheng, Jun Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2402.06378)] [[Code](https://github.com/zzr-idam/FDV-NET)]
- **MambaMIR: An Arbitrary-Masked Mamba for Joint Medical Image Reconstruction and Uncertainty Estimation.** [19 March, 2024] [ArXiv, 2024]<br/>
  *Jiahao Huang, Liutao Yang, Fanwen Wang, Yinzhe Wu, Yang Nan, Angelica I. Aviles-Rivero, Carola-Bibiane Schönlieb, Daoqiang Zhang, Guang Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2402.18451)] [[Code](https://github.com/ShumengLI/Mamba4MIS)]
- **FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion with Mamba.** [20 April, 2024] [ArXiv, 2024]<br/>
  *Xinyu Xie, Yawen Cui, Chio-In Ieong, Tao Tan, Xiaozhi Zhang, Xubin Zheng, Zitong Yu.*<br/>
  [[Paper](https://arxiv.org/abs/2404.09498)] [[Code](https://github.com/millieXie/FusionMamba)]
- **MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion.** [12 April, 2024] [ArXiv, 2024]<br/>
  *Zhe Li, Haiwei Pan, Kejia Zhang, Yuhua Wang, Fengming Yu.*<br/>
  [[Paper](https://arxiv.org/abs/2404.08406)]

#### 2.5 Other Medical Tasks

- **MD-Dose: A Diffusion Model based on the Mamba for Radiotherapy Dose Prediction.** [13 March, 2024] [ArXiv, 2024]<br/>
  *Linjie Fu, Xia Li, Xiuding Cai, Yingkai Wang, Xueyao Wang, Yali Shen, Yu Yao.*<br/>
  [[Paper](https://arxiv.org/abs/2403.08479)] [[Code](https://github.com/LinjieFu-U/mamba_dose)]
- **Motion-Guided Dual-Camera Tracker for Low-Cost Skill Evaluation of Gastric Endoscopy.** [20 April, 2024] [ArXiv, 2024]<br/>
  *Yuelin Zhang, Wanquan Yan, Kim Yan, Chun Ping Lam, Yufu Qiu, Pengyu Zheng, Raymond Shing-Yan Tang, Shing Shin Cheng.*<br/>
  [[Paper](https://arxiv.org/abs/2403.05146)] [[Code](https://github.com/PieceZhang/MotionDCTrack)]

## Other Domains

coming soon

