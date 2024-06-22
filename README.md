# Vision Mamba: A Comprehensive Survey and Taxonomy



> **Abstract:** *State Space Model (SSM) is a mathematical model used to describe and analyze the behavior of dynamic systems. This model has witnessed numerous applications in several fields, including control theory, signal processing, economics and machine learning. In the field of deep learning, state space models are used to process sequence data, such as time series analysis, natural language processing (NLP) and video understanding. By mapping sequence data to state space, long-term dependencies in the data can be better captured. In particular,  modern SSMs have shown strong representational capabilities in NLP, especially in long sequence modeling, while maintaining linear time complexity. Notably, based on the latest state-space models, Mamba \cite{Mamba} merges time-varying parameters into SSMs and formulates a hardware-aware algorithm for efficient training and inference. Given its impressive efficiency and strong long-range dependency modeling capability, Mamba is expected to become a new AI architecture that may outperform Transformer. Recently, a number of works have attempted to study the potential of Mamba in various fields, such as general vision, multi-modal, medical image analysis and remote sensing image analysis, by extending Mamba from natural language domain to visual domain. To fully understand Mamba in the visual domain, we conduct a comprehensive survey and present a taxonomy study. This survey focuses on Mamba's application to a variety of visual tasks and data types, and discusses its predecessors, recent advances and far-reaching impact on a wide range of domains. Since Mamba is now on an upward trend, please actively notice us if you have new findings, and new progress on Mamba will be included in this survey in a timely manner and updated on the website: (https://github.com/lx6c78/Vision-Mamba-A-Comprehensive-Survey-and-Taxonomy).*



:star: **We will timely update the latest representaive literatures and their released source code on this page. If you have any questions, please don't hesitate to contact us at any of the following emails:**
liuxiao@stu.cqu.edu.cn, zhangchenxu@cqu.edu.cn, leizhang@cqu.edu.cn



## 📢 Update Log

- 2024.05.07: Our paper is released! [[arXiv](https://arxiv.org/abs/2405.04404)]
- 2024.05.18: Added "Latest Visual Mamba Papers" column. We plan to update these papers in subsequent versions of our survey.



## Citation

If you find this repository is useful for you, please cite our paper:

```
@misc{liu2024vision,
      title={Vision Mamba: A Comprehensive Survey and Taxonomy}, 
      author={Xiao Liu and Chenxu Zhang and Lei Zhang},
      year={2024},
      eprint={2405.04404},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## Contents

- [Related Survey](#Related-Survey)
- [Latest vision Mamba paper](#Latest-vision-Mamba-paper)
- [General Vision](#General-Vision)
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
- **Computation-Efficient Era: A Comprehensive Survey of State Space Models in Medical Image Analysis.** [5 June, 2024] [ArXiv, 2024]<br/>
  *Moein Heidari, Sina Ghorbani Kolahi, Sanaz Karimijafarbigloo, Bobby Azad, Afshin Bozorgpour, Soheila Hatami, Reza Azad, Ali Diba, Ulas Bagci, Dorit Merhof, Ilker Hacihaliloglu.*<br/>
  [[Paper](https://arxiv.org/abs/2406.03430)] [[Gihub](https://github.com/xmindflow/Awesome_mamba)]



## Latest vision Mamba paper

> We plan to update these papers in subsequent versions of our survey.

* **CLIP-Mamba: CLIP Pretrained Mamba Models with OOD and Hessian Evaluation.** [30 April, 2024] [ArXiv, 2024]<br/>
  *Weiquan Huang, Yifei Shen, Yifan Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2404.19394)] [[Code](https://github.com/raytrun/mamba-clip)]
* **SOAR: Advancements in Small Body Object Detection for Aerial Imagery Using State Space Models and Programmable Gradients.** [5 May, 2024] [ArXiv, 2024]<br/>*Tushar Verma, Jyotsna Singh, Yash Bhartari, Rishi Jarwal, Suraj Singh, Shubhkarman Singh.*<br/>[[Paper](https://arxiv.org/abs/2405.01699)] [[Code](https://github.com/yash2629/S.O.A.R)] 
* **SSUMamba: Spatial-Spectral Selective State Space Model for Hyperspectral Image Denoising.** [15 May, 2024] [ArXiv, 2024]<br/>*Guanyiman Fu, Fengchao Xiong, Jianfeng Lu, Jun Zhou, Yuntao Qian.*<br/>
  [[Paper](https://arxiv.org/abs/2405.01726)] [[Code](https://github.com/lronkitty/SSUMamba)] 
* **FER-YOLO-Mamba: Facial Expression Detection and Classification Based on Selective State Space.** [9 May, 2024] [ArXiv, 2024]<br/>
  *Hui Ma, Sen Lei, Turgay Celik, Heng-Chao Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.01828)] [[Code](https://github.com/SwjtuMa/FER-YOLO-Mamba)] 
* **DVMSR: Distillated Vision Mamba for Efficient Super-Resolution.** [11 May, 2024] [ArXiv, 2024]<br/>
  *Xiaoyan Lei, Wenlong Zhang, Weifeng Cao.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03008)] [[Code](https://github.com/nathan66666/DVMSR)] 
* **AC-MAMBASEG: An adaptive convolution and Mamba-based architecture for enhanced skin lesion segmentation.** [5 May, 2024] [ArXiv, 2024]<br/>
  *Xiaoyan Lei, Wenlong Zhang, Weifeng Cao.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03011)] [[Code](https://github.com/vietthanh2710/AC-MambaSeg)] 
* **Retinexmamba: Retinex-based Mamba for Low-light Image Enhancement.** [6 May, 2024] [ArXiv, 2024]<br/>
  *Jiesong Bai, Yuhao Yin, Qiyuan He.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03349)] [[Code](https://github.com/YhuoyuH/RetinexMamba)] 
* **VMambaCC: A Visual State Space Model for Crowd Counting.** [6 May, 2024] [ArXiv, 2024]<br/>
  *Hao-Yuan Ma, Li Zhang, Shuai Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03978)] 
* **Traj-LLM: A New Exploration for Empowering Trajectory Prediction with Pre-trained Large Language Models.** [8 May, 2024] [ArXiv, 2024]<br/>
  *Zhengxing Lan, Hongbo Li, Lingshan Liu, Bo Fan, Yisheng Lv, Yilong Ren, Zhiyong Cui.*<br/>
  [[Paper](https://arxiv.org/abs/2405.04909)] 
* **Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution.** [8 May, 2024] [ArXiv, 2024]<br/>
  *Yi Xiao, Qiangqiang Yuan, Kui Jiang, Yuzeng Chen, Qiang Zhang, Chia-Wen Lin.*<br/>
  [[Paper](https://arxiv.org/abs/2405.04964)] 
* **HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation.** [11 May, 2024] [ArXiv, 2024]<br/>
  *Jiashu Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03349)] 
* **VM-DDPM: Vision Mamba Diffusion for Medical Image Synthesis.** [9 May, 2024] [ArXiv, 2024]<br/>
  *Zhihan Ju, Wanting Zhou.*<br/>
  [[Paper](https://arxiv.org/abs/2405.05667)] 
* **Rethinking Efficient and Effective Point-based Networks for Event Camera Classification and Regression: EventMamba.** [9 May, 2024] [ArXiv, 2024]<br/>
  *Hongwei Ren, Yue Zhou, Jiadong Zhu, Haotian Fu, Yulong Huang, Xiaopeng Lin, Yuetong Fang, Fei Ma, Hao Yu, Bojun Cheng.*<br/>
  [[Paper](https://arxiv.org/abs/2405.06116)] 
* **GMSR:Gradient-Guided Mamba for Spectral Reconstruction from RGB Images.** [13 May, 2024] [ArXiv, 2024]<br/>
  *Xinying Wang, Zhixiong Huang, Sifan Zhang, Jiawen Zhu, Lin Feng.*<br/>
  [[Paper](https://arxiv.org/abs/2405.07777)] [[Code](https://github.com/wxy11-27/GMSR)] 
* **OverlapMamba: Novel Shift State Space Model for LiDAR-based Place Recognition.** [13 May, 2024] [ArXiv, 2024]<br/>
  *Qiuchi Xiang, Jintao Cheng, Jiehao Luo, Jin Wu, Rui Fan, Xieyuanli Chen, Xiaoyu Tang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.07966)] 
* **MambaOut: Do We Really Need Mamba for Vision?** [14 May, 2024] [ArXiv, 2024]<br/>
  *Weihao Yu, Xinchao Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.07992)] [[Code](https://github.com/yuweihao/MambaOut)] 
* **Rethinking Scanning Strategies with Vision Mamba in Semantic Segmentation of Remote Sensing Imagery: An Experimental Study.** [14 May, 2024] [ArXiv, 2024]<br/>
  *Qinfeng Zhu, Yuan Fang, Yuanzhi Cai, Cheng Chen, Lei Fan.*<br/>
  [[Paper](https://arxiv.org/abs/2405.08493)] 
* **IRSRMamba: Infrared Image Super-Resolution via Mamba-based Wavelet Transform Feature Modulation Model.** [16 May, 2024] [ArXiv, 2024]<br/>
  *Yongsong Huang, Tomo Miyazaki, Xiaofeng Liu, Shinichiro Omachi.*<br/>
  [[Paper](https://arxiv.org/abs/2405.09873)] [[Code](https://github.com/yongsongH/IRSRMamba)] 
* **RSDehamba: Lightweight Vision Mamba for Remote Sensing Satellite Image Dehazing.** [16 May, 2024] [ArXiv, 2024]<br/>
  *Huiling Zhou, Xianhao Wu, Hongming Chen, Xiang Chen, Xin He.*<br/>
  [[Paper](https://arxiv.org/abs/2405.10030)] 
* **CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation.** [17 May, 2024] [ArXiv, 2024]<br/>
  *Mushui Liu, Jun Dan, Ziqian Lu, Yunlong Yu, Yingming Li, Xi Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.10530)] [[Code](https://github.com/XiaoBuL/CM-UNet)] 
* **Mamba-in-Mamba: Centralized Mamba-Cross-Scan in Tokenized Mamba Model for Hyperspectral Image Classification.** [25 May, 2024] [ArXiv, 2024]<br/>
  *Weilian Zhou, Sei-Ichiro Kamata, Haipeng Wang, Man-Sing Wong, Huiying, Hou.*<br/>
  [[Paper](https://arxiv.org/abs/2405.12003)] [[Code](https://github.com/zhouweilian1904/Mamba-in-Mamba)] 
* **3DSS-Mamba: 3D-Spectral-Spatial Mamba for Hyperspectral Image Classification.** [21 May, 2024] [ArXiv, 2024]<br/>
  *Yan He, Bing Tu, Bo Liu, Jun Li, Antonio Plaza.*<br/>
  [[Paper](https://arxiv.org/abs/2405.12487)] 
* **I2I-Mamba: Multi-modal medical image synthesis via selective state space modeling.** [22 May, 2024] [ArXiv, 2024]<br/>
  *Omer F. Atli, Bilal Kabas, Fuat Arslan, Mahmut Yurt, Onat Dalmaz, Tolga Çukur.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14022)] [[Code](https://github.com/icon-lab/I2I-Mamba)] 
* **Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model.** [23 May, 2024] [ArXiv, 2024]<br/>
  *Yuheng Shi, Minjing Dong, Chang Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14174)] [[Code](https://github.com/YuHengsss/MSVMamba)] 
* **DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis.** [23 May, 2024] [ArXiv, 2024]<br/>
  *Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, Xihui Liu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14224)] [[Code](https://github.com/tyshiwo1/DiM-DiffusionMamba/)] 
* **MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models.** [23 May, 2024] [ArXiv, 2024]<br/>
  *Jiuming Liu, Jinru Han, Lihao Liu, Angelica I. Aviles-Rivero, Chaokang Jiang, Zhe Liu, Hesheng Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14338)] 
* **Scalable Visual State Space Model with Fractal Scanning.** [26 May, 2024] [ArXiv, 2024]<br/>
  *Lv Tang, HaoKe Xiao, Peng-Tao Jiang, Hao Zhang, Jinwei Chen, Bo Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14480)] 
* **Mamba-R: Vision Mamba ALSO Needs Registers.** [23 May, 2024] [ArXiv, 2024]<br/>
  *Feng Wang, Jiahao Wang, Sucheng Ren, Guoyizhe Wei, Jieru Mei, Wei Shao, Yuyin Zhou, Alan Yuille, Cihang Xie.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14858)] [[Homepage](https://wangf3014.github.io/mambar-page/)] [[Code](https://github.com/wangf3014/Mamba-Reg)] 
* **PointRWKV: Efficient RWKV-Like Model for Hierarchical Point Cloud Learning.** [24 May, 2024] [ArXiv, 2024]<br/>
  *Qingdong He, Jiangning Zhang, Jinlong Peng, Haoyang He, Yabiao Wang, Chengjie Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15214)] [[Code](https://hithqd.github.io/projects/PointRWKV/)] 
* **PoinTramba: A Hybrid Transformer-Mamba Framework for Point Cloud Analysis.** [24 May, 2024] [ArXiv, 2024]<br/>
  *Zicheng Wang, Zhenghao Chen, Yiming Wu, Zhen Zhao, Luping Zhou, Dong Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15463)] [[Code](https://github.com/xiaoyao3302/PoinTramba)] 
* **Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Byung-Kwan Lee, Chae Won Kim, Beomchan Park, Yong Man Ro.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15574)] [[Code](https://github.com/ByungKwanLee/Meteor)] 
* **Scaling Diffusion Mamba with Bidirectional SSMs for Efficient Image and Video Generation.** [24 May, 2024] [ArXiv, 2024]<br/>
  *Shentong Mo, Yapeng Tian.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15881)] 
* **MUCM-Net: A Mamba Powered UCM-Net for Skin Lesion Segmentation.** [24 May, 2024] [ArXiv, 2024]<br/>
  *Chunyu Yuan, Dongfang Zhao, Sos S. Agaian.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15925)] [[Code](https://github.com/chunyuyuan/MUCM-Net)] 
* **Demystify Mamba in Vision: A Linear Attention Perspective.** [26 May, 2024] [ArXiv, 2024]<br/>
  *Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, Gao Huang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.16605)] [[Code](https://github.com/LeapLabTHU/MLLA)] 
* **TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Yinda Chen, Haoyuan Shi, Xiaoyu Liu, Te Shi, Ruobing Zhang, Dong Liu, Zhiwei Xiong, Feng Wu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.16847)] [[Code](https://github.com/ydchen0806/TokenUnify)] 
* **LCM: Locally Constrained Compact Point Cloud Model for Masked Point Modeling.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Yaohua Zha, Naiqi Li, Yanzi Wang, Tao Dai, Hang Guo, Bin Chen, Zhi Wang, Zhihao Ouyang, Shu-Tao Xia.*<br/>
  [[Paper](https://arxiv.org/abs/2405.17149)] 
* **Enhancing Global Sensitivity and Uncertainty Quantification in Medical Image Reconstruction with Monte Carlo Arbitrary-Masked Mamba.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Jiahao Huang, Liutao Yang, Fanwen Wang, Yinzhe Wu, Yang Nan, Weiwen Wu, Chengyan Wang, Kuangyu Shi, Angelica I. Aviles-Rivero, Carola-Bibiane Schönlieb, Daoqiang Zhang, Guang Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.17680)] 
* **Deciphering Movement: Unified Trajectory Generation Model for Multi-Agent.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Yi Xu, Yun Fu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.17680)] [[Code](https://github.com/colorfulfuture/UniTraj-pytorch)] 
* **DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention.** [28 May, 2024] [ArXiv, 2024]<br/>
  *Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, Xinggang Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.18428)] [[Code](https://github.com/hustvl/DiG)] 
* **Cardiovascular Disease Detection from Multi-View Chest X-rays with BI-Mamba.** [28 May, 2024] [ArXiv, 2024]<br/>
  *Zefan Yang, Jiajin Zhang, Ge Wang, Mannudeep K. Kalra, Pingkun Yan.*<br/>
  [[Paper](https://arxiv.org/abs/2405.18533)] 
* **Vim-F: Visual State Space Model Benefiting from Learning in the Frequency Domain.** [28 May, 2024] [ArXiv, 2024]<br/>
  *Juntao Zhang, Kun Bian, Peng Cheng, Wenbo An, Jianning Liu, Jun Zhou.*<br/>
  [[Paper](https://arxiv.org/abs/2405.18679)] [[Code](https://github.com/yws-wxs/Vim-F)] 
* **FourierMamba: Fourier Learning Integration with State Space Models for Image Deraining.** [29 May, 2024] [ArXiv, 2024]<br/>
  *Dong Li, Yidi Liu, Xueyang Fu, Senyan Xu, Zheng-Jun Zha.*<br/>
  [[Paper](https://arxiv.org/abs/2405.19450)] 
* **DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark.** [30 May, 2024] [ArXiv, 2024]<br/>
  *Haoxing Chen, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Yaohui Li, Jun Lan, Huijia Zhu, Jianfu Zhang, Weiqiang Wang, Huaxiong Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.19707)] [[Code](https://github.com/chenhaoxing/DeMamba)] 
* **Dual Hyperspectral Mamba for Efficient Spectral Compressive Imaging.** [1 June, 2024] [ArXiv, 2024]<br/>
  *Jiahua Dong, Hui Yin, Hongliu Li, Wenbo Li, Yulun Zhang, Salman Khan, Fahad Shahbaz Khan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.00449)] [[Code](https://github.com/JiahuaDong/DHM)] 
* **MGI: Multimodal Contrastive pre-training of Genomic and Medical Imaging.** [2 June, 2024] [ArXiv, 2024]<br/>
  *Jiaying Zhou, Mingzhou Jiang, Junde Wu, Jiayuan Zhu, Ziyue Wang, Yueming Jin.*<br/>
  [[Paper](https://arxiv.org/abs/2406.00631)] 
* **LLEMamba: Low-Light Enhancement via Relighting-Guided Mamba with Deep Unfolding Network.** [3 June, 2024] [ArXiv, 2024]<br/>
  *Xuanqi Zhang, Haijin Zeng, Jinwang Pan, Qiangqiang Shen, Yongyong Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2406.01028)] 
* **Dimba: Transformer-Mamba Diffusion Models.** [3 June, 2024] [ArXiv, 2024]<br/>
  *Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, Junshi Huang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.01159)] [[Homepage](https://dimba-project.github.io/)] [[Code](https://github.com/feizc/Dimba)] 
* **CDMamba: Remote Sensing Image Change Detection with Mamba.** [6 June, 2024] [ArXiv, 2024]<br/>
  *Haotian Zhang, Keyan Chen, Chenyang Liu, Hao Chen, Zhengxia Zou, Zhenwei Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2406.04207)] [[Code](https://github.com/zmoka-zht/CDMamba)] 
* **RoboMamba: Multimodal State Space Model for Efficient Robot Reasoning and Manipulation.** [6 June, 2024] [ArXiv, 2024]<br/>
  *Jiaming Liu, Mengzhen Liu, Zhenyu Wang, Lily Lee, Kaichen Zhou, Pengju An, Senqiao Yang, Renrui Zhang, Yandong Guo, Shanghang Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.04339)] [[Homepage](https://sites.google.com/view/robomamba-web)] [[Code](https://github.com/lmzpai/roboMamba)] 
* **MambaDepth: Enhancing Long-range Dependency for Self-Supervised Fine-Structured Monocular Depth Estimation.** [6 June, 2024] [ArXiv, 2024]<br/>
  *Ionuţ Grigore, Călin-Adrian Popa.*<br/>
  [[Paper](https://arxiv.org/abs/2406.04532)] [[Code](https://github.com/ionut-grigore99/MambaDepth)] 
* **Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs.** [7 June, 2024] [ArXiv, 2024]<br/>
  *Shentong Mo.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05038)]
* **HDMba: Hyperspectral Remote Sensing Imagery Dehazing with State Space Model.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Hang Fu, Genyun Sun, Yinhe Li, Jinchang Ren, Aizhu Zhang, Cheng Jing, Pedram Ghamisi.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05700)] [[Code](https://github.com/RsAI-lab/HDMba)]
* **Vision Mamba: Cutting-Edge Classification of Alzheimer's Disease with 3D MRI Scans.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Muthukumar K A, Amit Gurung, Priya Ranjan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05757)]
* **Convolution and Attention-Free Mamba-based Cardiac Image Segmentation.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Abbas Khan, Muhammad Asad, Martin Benning, Caroline Roney, Gregory Slabaugh.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05786)]
* **Mamba YOLO: SSMs-Based YOLO For Object Detection.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Zeyu Wang, Chen Li, Huiying Xu, Xinzhong Zhu.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05835)] [[Code](https://github.com/HZAI-ZJNU/Mamba-YOLO)]
* **MHS-VM: Multi-Head Scanning in Parallel Subspaces for Vision Mamba.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Zhongping Ji.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05992)] [[Code](https://github.com/PixDeep/MHS-VM)]
* **PointABM:Integrating Bidirectional State Space Model with Multi-Head Self-Attention for Point Cloud Analysis.** [10 June, 2024] [ArXiv, 2024]<br/>
  *Jia-wei Chen, Yu-jie Xiong, Yong-bin Gao.*<br/>
  [[Paper](https://arxiv.org/abs/2406.06069)]
* **DualMamba: A Lightweight Spectral-Spatial Mamba-Convolution Network for Hyperspectral Image Classification.** [11 June, 2024] [ArXiv, 2024]<br/>
  *Jiamu Sheng, Jingyi Zhou, Jiong Wang, Peng Ye, Jiayuan Fan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.07050)]
* **Autoregressive Pretraining with Mamba in Vision.** [11 June, 2024] [ArXiv, 2024]<br/>
  *Sucheng Ren, Xianhang Li, Haoqin Tu, Feng Wang, Fangxun Shu, Lei Zhang, Jieru Mei, Linjie Yang, Peng Wang, Heng Wang, Alan Yuille, Cihang Xie.*<br/>
  [[Paper](https://arxiv.org/abs/2406.07537)] [[Code](https://github.com/OliverRensu/ARM)]
* **PixMamba: Leveraging State Space Models in a Dual-Level Architecture for Underwater Image Enhancement.** [12 June, 2024] [ArXiv, 2024]<br/>
  *Wei-Tung Lin, Yong-Xiang Lin, Jyun-Wei Chen, Kai-Lung Hua.*<br/>
  [[Paper](https://arxiv.org/abs/2406.08444)] [[Code](https://github.com/weitunglin/pixmamba)]
* **On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models.** [12 June, 2024] [ArXiv, 2024]<br/>
  *Hashmat Shadab Malik, Numan Saeed, Asif Hanif, Muzammal Naseer, Mohammad Yaqub, Salman Khan, Fahad Shahbaz Khan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.08486)] [[Code](https://github.com/HashmatShadab/Robustness-of-Volumetric-Medical-Segmentation-Models)]
* **Q-Mamba: On First Exploration of Vision Mamba for Image Quality Assessment.** [13 June, 2024] [ArXiv, 2024]<br/>
  *Fengbin Guan, Xin Li, Zihao Yu, Yiting Lu, Zhibo Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2406.09546)]
* **Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection.** [18 June, 2024] [ArXiv, 2024]<br/>
  *Guowen Zhang, Lue Fan, Chenhang He, Zhen Lei, Zhaoxiang Zhang, Lei Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.10700)] [[Code](https://github.com/gwenzhang/Voxel-Mamba)]
* **PyramidMamba: Rethinking Pyramid Feature Fusion with Selective Space State Model for Semantic Segmentation of Remote Sensing Imagery.** [16 June, 2024] [ArXiv, 2024]<br/>
  *Libo Wang, Dongxu Li, Sijun Dong, Xiaoliang Meng, Xiaokang Zhang, Danfeng Hong.*<br/>
  [[Paper](https://arxiv.org/abs/2406.10828)] [[Code](https://github.com/WangLibo1995/GeoSeg)]
* **LFMamba: Light Field Image Super-Resolution with State Space Model.** [18 June, 2024] [ArXiv, 2024]<br/>
  *Wang xia, Yao Lu, Shunzhou Wang, Ziqi Wang, Peiqi Xia, Tianfei Zhou.*<br/>[[Paper](https://arxiv.org/abs/2406.12463)]

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
- **VMambaCC: A Visual State Space Model for Crowd Counting.** [6 May, 2024] [ArXiv, 2024]<br/>
  *Hao-Yuan Ma, Li Zhang, Shuai Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03978)]
- **MambaOut: Do We Really Need Mamba for Vision?** [14 May, 2024] [ArXiv, 2024]<br/>
  *Weihao Yu, Xinchao Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.07992)] [[Code](https://github.com/yuweihao/MambaOut)]
- **Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model.** [23 May, 2024] [ArXiv, 2024]<br/>
  *Yuheng Shi, Minjing Dong, Chang Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14174)] [[Code](https://github.com/YuHengsss/MSVMamba)]
- **Mamba-R: Vision Mamba ALSO Needs Registers.** [23 May, 2024] [ArXiv, 2024]<br/>
  *Feng Wang, Jiahao Wang, Sucheng Ren, Guoyizhe Wei, Jieru Mei, Wei Shao, Yuyin Zhou, Alan Yuille, Cihang Xie.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14858)] [[Homepage](https://wangf3014.github.io/mambar-page/)] [[Code](https://github.com/wangf3014/Mamba-Reg)]
- **Demystify Mamba in Vision: A Linear Attention Perspective.** [26 May, 2024] [ArXiv, 2024]<br/>
  *Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, Gao Huang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.16605)] [[Code](https://github.com/LeapLabTHU/MLLA)]
- **Vim-F: Visual State Space Model Benefiting from Learning in the Frequency Domain.** [28 May, 2024] [ArXiv, 2024]<br/>
  *Juntao Zhang, Kun Bian, Peng Cheng, Wenbo An, Jianning Liu, Jun Zhou.*<br/>
  [[Paper](https://arxiv.org/abs/2405.18679)] [[Code](https://github.com/yws-wxs/Vim-F)]
- **Mamba YOLO: SSMs-Based YOLO For Object Detection.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Zeyu Wang, Chen Li, Huiying Xu, Xinzhong Zhu.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05835)] [[Code](https://github.com/HZAI-ZJNU/Mamba-YOLO)]
- **Autoregressive Pretraining with Mamba in Vision.** [11 June, 2024] [ArXiv, 2024]<br/>
  *Sucheng Ren, Xianhang Li, Haoqin Tu, Feng Wang, Fangxun Shu, Lei Zhang, Jieru Mei, Linjie Yang, Peng Wang, Heng Wang, Alan Yuille, Cihang Xie.*<br/>
  [[Paper](https://arxiv.org/abs/2406.07537)] [[Code](https://github.com/OliverRensu/ARM)]

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
- **MemoryMamba: Memory-Augmented State Space Model for Defect Recognition.** [6 May, 2024] [ArXiv, 2024] <br/>
  *Qianning Wang, He Hu, Yucheng Zhou.*<br/>[[Paper](https://arxiv.org/abs/2405.03673)]
- **SOAR: Advancements in Small Body Object Detection for Aerial Imagery Using State Space Models and Programmable Gradients.** [5 May, 2024] [ArXiv, 2024]<br/>*Tushar Verma, Jyotsna Singh, Yash Bhartari, Rishi Jarwal, Suraj Singh, Shubhkarman Singh.*<br/>[[Paper](https://arxiv.org/abs/2405.01699)] [[Code](https://github.com/yash2629/S.O.A.R)]
- **FER-YOLO-Mamba: Facial Expression Detection and Classification Based on Selective State Space.** [9 May, 2024] [ArXiv, 2024]<br/>
  *Hui Ma, Sen Lei, Turgay Celik, Heng-Chao Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.01828)] [[Code](https://github.com/SwjtuMa/FER-YOLO-Mamba)]
- **OverlapMamba: Novel Shift State Space Model for LiDAR-based Place Recognition.** [13 May, 2024] [ArXiv, 2024]<br/>
  *Qiuchi Xiang, Jintao Cheng, Jiehao Luo, Jin Wu, Rui Fan, Xieyuanli Chen, Xiaoyu Tang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.07966)]
- **TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Yinda Chen, Haoyuan Shi, Xiaoyu Liu, Te Shi, Ruobing Zhang, Dong Liu, Zhiwei Xiong, Feng Wu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.16847)] [[Code](https://github.com/ydchen0806/TokenUnify)]
- **DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark.** [30 May, 2024] [ArXiv, 2024]<br/>
  *Haoxing Chen, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Yaohui Li, Jun Lan, Huijia Zhu, Jianfu Zhang, Weiqiang Wang, Huaxiong Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.19707)] [[Code](https://github.com/chenhaoxing/DeMamba)]
- **MambaDepth: Enhancing Long-range Dependency for Self-Supervised Fine-Structured Monocular Depth Estimation.** [6 June, 2024] [ArXiv, 2024]<br/>
  *Ionuţ Grigore, Călin-Adrian Popa.*<br/>
  [[Paper](https://arxiv.org/abs/2406.04532)] [[Code](https://github.com/ionut-grigore99/MambaDepth)]
- **Q-Mamba: On First Exploration of Vision Mamba for Image Quality Assessment.** [13 June, 2024] [ArXiv, 2024]<br/>
  *Fengbin Guan, Xin Li, Zihao Yu, Yiting Lu, Zhibo Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2406.09546)]
- **Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection.** [18 June, 2024] [ArXiv, 2024]<br/>
  *Guowen Zhang, Lue Fan, Chenhang He, Zhen Lei, Zhaoxiang Zhang, Lei Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.10700)] [[Code](https://github.com/gwenzhang/Voxel-Mamba)]

### 2 Low-level Vision

#### 2.1 Image Denoising and Enhancement

- **U-shaped Vision Mamba for Single Image Dehazing.** [15 February, 2024] [ArXiv, 2024]<br/>
  *Zhuoran Zheng, Chen Wu.*<br/>[[Paper](https://arxiv.org/abs/2402.04139)] [[Code](https://github.com/zzr-idam)]
- **FreqMamba: Viewing Mamba from a Frequency Perspective for Image Deraining.** [15 April, 2024] [ArXiv, 2024]<br/>
  *Zou Zhen, Yu Hu, Zhao Feng.*<br/>[[Paper](https://arxiv.org/abs/2404.09476)]
- **Retinexmamba: Retinex-based Mamba for Low-light Image Enhancement.** [6 May, 2024] [ArXiv, 2024]<br/>
  *Jiesong Bai, Yuhao Yin, Qiyuan He.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03349)] [[Code](https://github.com/YhuoyuH/RetinexMamba)]
- **HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation.** [11 May, 2024] [ArXiv, 2024]<br/>
  *Jiashu Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03349)]
- **FourierMamba: Fourier Learning Integration with State Space Models for Image Deraining.** [29 May, 2024] [ArXiv, 2024]<br/>
  *Dong Li, Yidi Liu, Xueyang Fu, Senyan Xu, Zheng-Jun Zha.*<br/>
  [[Paper](https://arxiv.org/abs/2405.19450)]
- **LLEMamba: Low-Light Enhancement via Relighting-Guided Mamba with Deep Unfolding Network.** [3 June, 2024] [ArXiv, 2024]<br/>
  *Xuanqi Zhang, Haijin Zeng, Jinwang Pan, Qiangqiang Shen, Yongyong Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2406.01028)]
- **PixMamba: Leveraging State Space Models in a Dual-Level Architecture for Underwater Image Enhancement.** [12 June, 2024] [ArXiv, 2024]<br/>
  *Wei-Tung Lin, Yong-Xiang Lin, Jyun-Wei Chen, Kai-Lung Hua.*<br/>
  [[Paper](https://arxiv.org/abs/2406.08444)] [[Code](https://github.com/weitunglin/pixmamba)]

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
- **Retinexmamba: Retinex-based Mamba for Low-light Image Enhancement.** [6 May, 2024] [ArXiv, 2024]<br/>
  *Jiesong Bai, Yuhao Yin, Qiyuan He.*<br/>[[Paper](https://arxiv.org/abs/2405.03349)] [[Code](https://github.com/YhuoyuH/RetinexMamba)]
- **DVMSR: Distillated Vision Mamba for Efficient Super-Resolution.** [11 May, 2024] [ArXiv, 2024]<br/>
  *Xiaoyan Lei, Wenlong Zhang, Weifeng Cao.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03008)] [[Code](https://github.com/nathan66666/DVMSR)]
- **IRSRMamba: Infrared Image Super-Resolution via Mamba-based Wavelet Transform Feature Modulation Model.** [16 May, 2024] [ArXiv, 2024]<br/>
  *Yongsong Huang, Tomo Miyazaki, Xiaofeng Liu, Shinichiro Omachi.*<br/>
  [[Paper](https://arxiv.org/abs/2405.09873)] [[Code](https://github.com/yongsongH/IRSRMamba)]
- **Scalable Visual State Space Model with Fractal Scanning.** [26 May, 2024] [ArXiv, 2024]<br/>
  *Lv Tang, HaoKe Xiao, Peng-Tao Jiang, Hao Zhang, Jinwei Chen, Bo Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14480)]
- **GMSR:Gradient-Guided Mamba for Spectral Reconstruction from RGB Images.** [13 May, 2024] [ArXiv, 2024]<br/>
  *Xinying Wang, Zhixiong Huang, Sifan Zhang, Jiawen Zhu, Lin Feng.*<br/>
  [[Paper](https://arxiv.org/abs/2405.07777)] [[Code](https://github.com/wxy11-27/GMSR)]
- **Dual Hyperspectral Mamba for Efficient Spectral Compressive Imaging.** [1 June, 2024] [ArXiv, 2024]<br/>
  *Jiahua Dong, Hui Yin, Hongliu Li, Wenbo Li, Yulun Zhang, Salman Khan, Fahad Shahbaz Khan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.00449)] [[Code](https://github.com/JiahuaDong/DHM)]
- **LFMamba: Light Field Image Super-Resolution with State Space Model.** [18 June, 2024] [ArXiv, 2024]<br/>
  *Wang xia, Yao Lu, Shunzhou Wang, Ziqi Wang, Peiqi Xia, Tianfei Zhou.*<br/>[[Paper](https://arxiv.org/abs/2406.12463)]

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
- **Rethinking Efficient and Effective Point-based Networks for Event Camera Classification and Regression: EventMamba.** [9 May, 2024] [ArXiv, 2024]<br/>
  *Hongwei Ren, Yue Zhou, Jiadong Zhu, Haotian Fu, Yulong Huang, Xiaopeng Lin, Yuetong Fang, Fei Ma, Hao Yu, Bojun Cheng.*<br/>
  [[Paper](https://arxiv.org/abs/2405.06116)]
- **MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models.** [23 May, 2024] [ArXiv, 2024]<br/>
  *Jiuming Liu, Jinru Han, Lihao Liu, Angelica I. Aviles-Rivero, Chaokang Jiang, Zhe Liu, Hesheng Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14338)]
- **PointRWKV: Efficient RWKV-Like Model for Hierarchical Point Cloud Learning.** [24 May, 2024] [ArXiv, 2024]<br/>
  *Qingdong He, Jiangning Zhang, Jinlong Peng, Haoyang He, Yabiao Wang, Chengjie Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15214)] [[Code](https://hithqd.github.io/projects/PointRWKV/)]
- **PoinTramba: A Hybrid Transformer-Mamba Framework for Point Cloud Analysis.** [24 May, 2024] [ArXiv, 2024]<br/>
  *Zicheng Wang, Zhenghao Chen, Yiming Wu, Zhen Zhao, Luping Zhou, Dong Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15463)] [[Code](https://github.com/xiaoyao3302/PoinTramba)]
- **LCM: Locally Constrained Compact Point Cloud Model for Masked Point Modeling.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Yaohua Zha, Naiqi Li, Yanzi Wang, Tao Dai, Hang Guo, Bin Chen, Zhi Wang, Zhihao Ouyang, Shu-Tao Xia.*<br/>
  [[Paper](https://arxiv.org/abs/2405.17149)]
- **Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs.** [7 June, 2024] [ArXiv, 2024]<br/>
  *Shentong Mo.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05038)]
- **PointABM:Integrating Bidirectional State Space Model with Multi-Head Self-Attention for Point Cloud Analysis.** [10 June, 2024] [ArXiv, 2024]<br/>
  *Jia-wei Chen, Yu-jie Xiong, Yong-bin Gao.*<br/>
  [[Paper](https://arxiv.org/abs/2406.06069)]

#### 3.2 Hyperspectral Imaging Analysis

- **Mamba-FETrack: Frame-Event Tracking via State Space Model.** [28 April, 2024] [ArXiv, 2024]<br/>
  *Ju Huang, Shiao Wang, Shuai Wang, Zhe Wu, Xiao Wang, Bo Jiang.*<br/>
  [[Paper](https://arxiv.org/abs/2404.18174)] [[Code](https://github.com/Event-AHU/Mamba_FETrack)]
- **3DSS-Mamba: 3D-Spectral-Spatial Mamba for Hyperspectral Image Classification.** [21 May, 2024] [ArXiv, 2024]<br/>
  *Yan He, Bing Tu, Bo Liu, Jun Li, Antonio Plaza.*<br/>
  [[Paper](https://arxiv.org/abs/2405.12487)]
- **DualMamba: A Lightweight Spectral-Spatial Mamba-Convolution Network for Hyperspectral Image Classification.** [11 June, 2024] [ArXiv, 2024]<br/>
  *Jiamu Sheng, Jingyi Zhou, Jiong Wang, Peng Ye, Jiayuan Fan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.07050)]

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
- **SMCD: High Realism Motion Style Transfer via Mamba-based Diffusion.** [5 May, 2024] [ArXiv, 2024]<br/>
  *Ziyun Qian, Zeyu Xiao, Zhenyi Wu, Dingkang Yang, Mingcheng Li, Shunli Wang, Shuaibing Wang, Dongliang Kou, Lihua Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.02844)]
- **DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis.** [23 May, 2024] [ArXiv, 2024]<br/>
  *Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, Xihui Liu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14224)] [[Code](https://github.com/tyshiwo1/DiM-DiffusionMamba/)]
- **Scaling Diffusion Mamba with Bidirectional SSMs for Efficient Image and Video Generation.** [24 May, 2024] [ArXiv, 2024]<br/>
  *Shentong Mo, Yapeng Tian.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15881)]
- **Deciphering Movement: Unified Trajectory Generation Model for Multi-Agent.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Yi Xu, Yun Fu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.17680)] [[Code](https://github.com/colorfulfuture/UniTraj-pytorch)]
- **DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention.** [28 May, 2024] [ArXiv, 2024]<br/>
  *Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, Xinggang Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.18428)] [[Code](https://github.com/hustvl/DiG)]
- **Dimba: Transformer-Mamba Diffusion Models.** [3 June, 2024] [ArXiv, 2024]<br/>
  *Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, Junshi Huang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.01159)] [[Homepage](https://dimba-project.github.io/)] [[Code](https://github.com/feizc/Dimba)]


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
- **CLIP-Mamba: CLIP Pretrained Mamba Models with OOD and Hessian Evaluation.** [30 April, 2024] [ArXiv, 2024]<br/>
  *Weiquan Huang, Yifei Shen, Yifan Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2404.19394)] [[Code](https://github.com/raytrun/mamba-clip)]
- **Traj-LLM: A New Exploration for Empowering Trajectory Prediction with Pre-trained Large Language Models.** [8 May, 2024] [ArXiv, 2024]<br/>
  *Zhengxing Lan, Hongbo Li, Lingshan Liu, Bo Fan, Yisheng Lv, Yilong Ren, Zhiyong Cui.*<br/>
  [[Paper](https://arxiv.org/abs/2405.04909)]
- **Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Byung-Kwan Lee, Chae Won Kim, Beomchan Park, Yong Man Ro.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15574)] [[Code](https://github.com/ByungKwanLee/Meteor)]
- **RoboMamba: Multimodal State Space Model for Efficient Robot Reasoning and Manipulation.** [6 June, 2024] [ArXiv, 2024]<br/>
  *Jiaming Liu, Mengzhen Liu, Zhenyu Wang, Lily Lee, Kaichen Zhou, Pengju An, Senqiao Yang, Renrui Zhang, Yandong Guo, Shanghang Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.04339)] [[Homepage](https://sites.google.com/view/robomamba-web)] [[Code](https://github.com/lmzpai/roboMamba)]

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
- **SSUMamba: Spatial-Spectral Selective State Space Model for Hyperspectral Image Denoising.** [15 May, 2024] [ArXiv, 2024]<br/>*Guanyiman Fu, Fengchao Xiong, Jianfeng Lu, Jun Zhou, Yuntao Qian.*<br/>
  [[Paper](https://arxiv.org/abs/2405.01726)] [[Code](https://github.com/lronkitty/SSUMamba)]
- **Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution.** [8 May, 2024] [ArXiv, 2024]<br/>
  *Yi Xiao, Qiangqiang Yuan, Kui Jiang, Yuzeng Chen, Qiang Zhang, Chia-Wen Lin.*<br/>
  [[Paper](https://arxiv.org/abs/2405.04964)]
- **RSDehamba: Lightweight Vision Mamba for Remote Sensing Satellite Image Dehazing.** [16 May, 2024] [ArXiv, 2024]<br/>
  *Huiling Zhou, Xianhao Wu, Hongming Chen, Xiang Chen, Xin He.*<br/>
  [[Paper](https://arxiv.org/abs/2405.10030)]
- **HDMba: Hyperspectral Remote Sensing Imagery Dehazing with State Space Model.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Hang Fu, Genyun Sun, Yinhe Li, Jinchang Ren, Aizhu Zhang, Cheng Jing, Pedram Ghamisi.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05700)] [[Code](https://github.com/RsAI-lab/HDMba)]

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
- **Mamba-in-Mamba: Centralized Mamba-Cross-Scan in Tokenized Mamba Model for Hyperspectral Image Classification.** [25 May, 2024] [ArXiv, 2024]<br/>
  *Weilian Zhou, Sei-Ichiro Kamata, Haipeng Wang, Man-Sing Wong, Huiying, Hou.*<br/>
  [[Paper](https://arxiv.org/abs/2405.12003)] [[Code](https://github.com/zhouweilian1904/Mamba-in-Mamba)]

#### 1.3 Remote Sensing Image Change Detection

- **ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model.** [14 April, 2024] [ArXiv, 2024]<br/>
  *Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, Naoto Yokoya.*<br/>
  [[Paper](https://arxiv.org/abs/2404.03425)] [[Code](https://github.com/ChenHongruixuan/MambaCD)]
- **RSCaMa: Remote Sensing Image Change Captioning with State Space Model.** [2 May, 2024] [ArXiv, 2024]<br/>
  *Chenyang Liu, Keyan Chen, Bowen Chen, Haotian Zhang, Zhengxia Zou, Zhenwei Shi.*<br/>[[Paper](https://arxiv.org/abs/2404.18895)] [[Code](https://github.com/Chen-Yang-Liu/RSCaMa)]
- **CDMamba: Remote Sensing Image Change Detection with Mamba.** [6 June, 2024] [ArXiv, 2024]<br/>
  *Haotian Zhang, Keyan Chen, Chenyang Liu, Hao Chen, Zhengxia Zou, Zhenwei Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2406.04207)] [[Code](https://github.com/zmoka-zht/CDMamba)]

#### 1.4 Remote Sensing Image Segmentation

- **Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model.** [11 April, 2024] [ArXiv, 2024]<br/>
  *Qinfeng Zhu, Yuanzhi Cai, Yuan Fang, Yihan Yang, Cheng Chen, Lei Fan, Anh Nguyen.*<br>
  [[Paper](https://arxiv.org/abs/2404.01705)] [[Code](https://github.com/zhuqinfeng1999/Samba)]
- **RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation.** [3 April, 2024] [ArXiv, 2024]<br/>
  *Xianping Ma, Xiaokang Zhang, Man-On Pun.*<br/>
  [[Paper](https://arxiv.org/abs/2404.02457)] [[Code](https://github.com/sstary/SSRS)]
- **RS-Mamba for Large Remote Sensing Image Dense Prediction.** [10 April, 2024] [ArXiv, 2024]<br/>
  *Sijie Zhao, Hao Chen, Xueliang Zhang, Pengfeng Xiao, Lei Bai, Wanli Ouyang.*<br/>[[Paper](https://arxiv.org/abs/2404.02668)] [[Code](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba)]
- **Rethinking Scanning Strategies with Vision Mamba in Semantic Segmentation of Remote Sensing Imagery: An Experimental Study.** [14 May, 2024] [ArXiv, 2024]<br/>
  *Qinfeng Zhu, Yuan Fang, Yuanzhi Cai, Cheng Chen, Lei Fan.*<br/>
  [[Paper](https://arxiv.org/abs/2405.08493)]
- **CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation.** [17 May, 2024] [ArXiv, 2024]<br/>
  *Mushui Liu, Jun Dan, Ziqian Lu, Yunlong Yu, Yingming Li, Xi Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.10530)] [[Code](https://github.com/XiaoBuL/CM-UNet)]
- **PyramidMamba: Rethinking Pyramid Feature Fusion with Selective Space State Model for Semantic Segmentation of Remote Sensing Imagery.** [16 June, 2024] [ArXiv, 2024]<br/>
  *Libo Wang, Dongxu Li, Sijun Dong, Xiaoliang Meng, Xiaokang Zhang, Danfeng Hong.*<br/>
  [[Paper](https://arxiv.org/abs/2406.10828)] [[Code](https://github.com/WangLibo1995/GeoSeg)]

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
- **AC-MAMBASEG: An adaptive convolution and Mamba-based architecture for enhanced skin lesion segmentation.** [5 May, 2024] [ArXiv, 2024]<br/>
  *Xiaoyan Lei, Wenlong Zhang, Weifeng Cao.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03011)] [[Code](https://github.com/vietthanh2710/AC-MambaSeg)]
- **MUCM-Net: A Mamba Powered UCM-Net for Skin Lesion Segmentation.** [24 May, 2024] [ArXiv, 2024]<br/>
  *Chunyu Yuan, Dongfang Zhao, Sos S. Agaian.*<br/>
  [[Paper](https://arxiv.org/abs/2405.15925)] [[Code](https://github.com/chunyuyuan/MUCM-Net)]
- **Convolution and Attention-Free Mamba-based Cardiac Image Segmentation.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Abbas Khan, Muhammad Asad, Martin Benning, Caroline Roney, Gregory Slabaugh.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05786)]
- **MHS-VM: Multi-Head Scanning in Parallel Subspaces for Vision Mamba.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Zhongping Ji.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05992)] [[Code](https://github.com/PixDeep/MHS-VM)]

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
- **Cardiovascular Disease Detection from Multi-View Chest X-rays with BI-Mamba.** [28 May, 2024] [ArXiv, 2024]<br/>
  *Zefan Yang, Jiajin Zhang, Ge Wang, Mannudeep K. Kalra, Pingkun Yan.*<br/>
  [[Paper](https://arxiv.org/abs/2405.18533)]
- **MGI: Multimodal Contrastive pre-training of Genomic and Medical Imaging.** [2 June, 2024] [ArXiv, 2024]<br/>
  *Jiaying Zhou, Mingzhou Jiang, Junde Wu, Jiayuan Zhu, Ziyue Wang, Yueming Jin.*<br/>
  [[Paper](https://arxiv.org/abs/2406.00631)]
- **Vision Mamba: Cutting-Edge Classification of Alzheimer's Disease with 3D MRI Scans.** [9 June, 2024] [ArXiv, 2024]<br/>
  *Muthukumar K A, Amit Gurung, Priya Ranjan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.05757)]

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
- **Enhancing Global Sensitivity and Uncertainty Quantification in Medical Image Reconstruction with Monte Carlo Arbitrary-Masked Mamba.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Jiahao Huang, Liutao Yang, Fanwen Wang, Yinzhe Wu, Yang Nan, Weiwen Wu, Chengyan Wang, Kuangyu Shi, Angelica I. Aviles-Rivero, Carola-Bibiane Schönlieb, Daoqiang Zhang, Guang Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.17680)]

#### 2.5 Other Medical Tasks

- **MD-Dose: A Diffusion Model based on the Mamba for Radiotherapy Dose Prediction.** [13 March, 2024] [ArXiv, 2024]<br/>
  *Linjie Fu, Xia Li, Xiuding Cai, Yingkai Wang, Xueyao Wang, Yali Shen, Yu Yao.*<br/>
  [[Paper](https://arxiv.org/abs/2403.08479)] [[Code](https://github.com/LinjieFu-U/mamba_dose)]
- **Motion-Guided Dual-Camera Tracker for Low-Cost Skill Evaluation of Gastric Endoscopy.** [20 April, 2024] [ArXiv, 2024]<br/>
  *Yuelin Zhang, Wanquan Yan, Kim Yan, Chun Ping Lam, Yufu Qiu, Pengyu Zheng, Raymond Shing-Yan Tang, Shing Shin Cheng.*<br/>
  [[Paper](https://arxiv.org/abs/2403.05146)] [[Code](https://github.com/PieceZhang/MotionDCTrack)]
- **VM-DDPM: Vision Mamba Diffusion for Medical Image Synthesis.** [9 May, 2024] [ArXiv, 2024]<br/>
  *Zhihan Ju, Wanting Zhou.*<br/>
  [[Paper](https://arxiv.org/abs/2405.05667)]
- **I2I-Mamba: Multi-modal medical image synthesis via selective state space modeling.** [22 May, 2024] [ArXiv, 2024]<br/>
  *Omer F. Atli, Bilal Kabas, Fuat Arslan, Mahmut Yurt, Onat Dalmaz, Tolga Çukur.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14022)] [[Code](https://github.com/icon-lab/I2I-Mamba)]
- **On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models.** [12 June, 2024] [ArXiv, 2024]<br/>
  *Hashmat Shadab Malik, Numan Saeed, Asif Hanif, Muzammal Naseer, Mohammad Yaqub, Salman Khan, Fahad Shahbaz Khan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.08486)] [[Code](https://github.com/HashmatShadab/Robustness-of-Volumetric-Medical-Segmentation-Models)]

## Other Domains

coming soon
