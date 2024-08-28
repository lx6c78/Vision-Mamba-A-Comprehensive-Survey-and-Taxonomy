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
* **MambaST: A Plug-and-Play Cross-Spectral Spatial-Temporal Fuser for Efficient Pedestrian Detection.** [2 August, 2024] [ArXiv, 2024]<br/>
  *Xiangbo Gao, Asiegbu Miracle Kanu-Asiegbu, Xiaoxiao Du.*<br/>
  [[Paper](https://arxiv.org/abs/2408.01037)] [[Code](https://github.com/XiangboGaoBarry/MambaST)]
* **Multi-head Spatial-Spectral Mamba for Hyperspectral Image Classification.** [26 August, 2024] [ArXiv, 2024]<br/>
  *Muhammad Ahmad, Muhammad Hassaan Farooq Butt, Muhammad Usama, Hamad Ahmed Altuwaijri, Manuel Mazzara, Salvatore Distefano.*<br/>
  [[Paper](https://arxiv.org/abs/2408.01224)]
* **WaveMamba: Spatial-Spectral Wavelet Mamba for Hyperspectral Image Classification.** [2 August, 2024] [ArXiv, 2024]<br/>
  *Muhammad Ahmad, Muhammad Usama, Manual Mazzara.*<br/>
  [[Paper](https://arxiv.org/abs/2408.01231)]
* **Wave-Mamba: Wavelet State Space Model for Ultra-High-Definition Low-Light Image Enhancement.** [2 August, 2024] [ArXiv, 2024]<br/>
  *Wenbin Zou, Hongxia Gao, Weipeng Yang, Tongtong Liu.*<br/>
  [[Paper](https://arxiv.org/abs/2408.01276)] [[Code](https://github.com/AlexZou14/Wave-Mamba)]
* **Spatial-Spectral Morphological Mamba for Hyperspectral Image Classification.** [23 August, 2024] [ArXiv, 2024]<br/>
  *Muhammad Ahmad, Muhammad Hassaan Farooq Butt, Muhammad Usama, Adil Mehmood Khan, Manuel Mazzara, Salvatore Distefano, Hamad Ahmed Altuwaijri, Swalpa Kumar Roy, Jocelyn Chanussot, Danfeng Hong.*<br/>
  [[Paper](https://arxiv.org/abs/2408.01372)]
* **JambaTalk: Speech-Driven 3D Talking Head Generation Based on Hybrid Transformer-Mamba Model.** [2 August, 2024] [ArXiv, 2024]<br/>
  *Farzaneh Jafari, Stefano Berretti, Anup Basu.*<br/>
  [[Paper](https://arxiv.org/abs/2408.01627)]
* **DeMansia: Mamba Never Forgets Any Tokens.** [4 August, 2024] [ArXiv, 2024]<br/>
  *Ricky Fang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.01986)] [[Code](https://github.com/catalpaaa/DeMansia)]
* **BioMamba: A Pre-trained Biomedical Language Representation Model Leveraging Mamba.** [5 August, 2024] [ArXiv, 2024]<br/>
  *Ling Yue, Sixue Xing, Yingzhou Lu, Tianfan Fu.*<br/>
  [[Paper](https://arxiv.org/abs/2408.02600)]
* **LaMamba-Diff: Linear-Time High-Fidelity Diffusion Models Based on Local Attention and Mamba.** [5 August, 2024] [ArXiv, 2024]<br/>
  *Yunxiang Fu, Chaoqi Chen, Yizhou Yu.*<br/>
  [[Paper](https://arxiv.org/abs/2408.02615)]
* **Context-aware Mamba-based Reinforcement Learning for social robot navigation.** [5 August, 2024] [ArXiv, 2024]<br/>
  *Syed Muhammad Mustafa, Omema Rizvi, Zain Ahmed Usmani, Abdul Basit Memon.*<br/>
  [[Paper](https://arxiv.org/abs/2408.02661)]
* **Pose Magic: Efficient and Temporally Consistent Human Pose Estimation with a Hybrid Mamba-GCN Network.** [7 August, 2024] [ArXiv, 2024]<br/>
  *Xinyi Zhang, Qiqi Bao, Qinpeng Cui, Wenming Yang, Qingmin Liao.*<br/>
  [[Paper](https://arxiv.org/abs/2408.02922)]
* **PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model.** [7 August, 2024] [ArXiv, 2024]<br/>
  *Yunlong Huang, Junshuo Liu, Ke Xian, Robert Caiming Qiu.*<br/>
  [[Paper](https://arxiv.org/abs/2408.03540)]
* **Neural Architecture Search based Global-local Vision Mamba for Palm-Vein Recognition.** [13 August, 2024] [ArXiv, 2024]<br/>
  *Huafeng Qin, Yuming Fu, Jing Chen, Mounim A. El-Yacoubi, Xinbo Gao, Jun Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.05743)]
* **Costal Cartilage Segmentation with Topology Guided Deformable Mamba: Method and Benchmark.** [14 August, 2024] [ArXiv, 2024]<br/>
  *Senmao Wang, Haifan Gong, Runmeng Cui, Boyao Wan, Yicheng Liu, Zhonglin Hu, Haiqing Yang, Jingyang Zhou, Bo Pan, Lin Lin, Haiyue Jiang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.07444)]
* **MambaVT: Spatio-Temporal Contextual Modeling for robust RGB-T Tracking.** [14 August, 2024] [ArXiv, 2024]<br/>
  *Simiao Lai, Chang Liu, Jiawen Zhu, Ben Kang, Yang Liu, Dong Wang, Huchuan Lu.*<br/>
  [[Paper](https://arxiv.org/abs/2408.07889)]
* **MambaMIM: Pre-training Mamba with State Space Token-interpolation.** [15 August, 2024] [ArXiv, 2024]<br/>
  *Fenghe Tang, Bingkun Nian, Yingtai Li, Jie Yang, Liu Wei, S. Kevin Zhou.*<br/>
  [[Paper](https://arxiv.org/abs/2408.08070)] [[Code](https://github.com/FengheTan9/MambaMIM)]
* **ColorMamba: Towards High-quality NIR-to-RGB Spectral Translation with Mamba.** [15 August, 2024] [ArXiv, 2024]<br/>
  *Huiyu Zhai, Guang Jin, Xingxing Yang, Guosheng Kang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.08087)] [[Code](https://github.com/AlexYangxx/ColorMamba/)]
* **QMambaBSR: Burst Image Super-Resolution with Query State Space Model.** [16 August, 2024] [ArXiv, 2024]<br/>
  *Xin Di, Long Peng, Peizhe Xia, Wenbo Li, Renjing Pei, Yang Cao, Yang Wang, Zheng-Jun Zha.*<br/>
  [[Paper](https://arxiv.org/abs/2408.08665)]
* **RGBT Tracking via All-layer Multimodal Interactions with Progressive Fusion Mamba.** [16 August, 2024] [ArXiv, 2024]<br/>
  *Andong Lu, Wanyu Wang, Chenglong Li, Jin Tang, Bin Luo.*<br/>
  [[Paper](https://arxiv.org/abs/2408.08827)]
* **MambaTrack: A Simple Baseline for Multiple Object Tracking with State Space Model.** [17 August, 2024] [ArXiv, 2024]<br/>
  *Changcheng Xiao, Qiong Cao, Zhigang Luo, Long Lan.*<br/>
  [[Paper](https://arxiv.org/abs/2408.09178)]
* **R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation.** [19 August, 2024] [ArXiv, 2024]<br/>
  *Xiao Wang, Yuehang Li, Fuling Wang, Shiao Wang, Chuanfu Li, Bo Jiang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.09743)] [[Code](https://github.com/Event-AHU/Medical_Image_Analysis)]
* **Event Stream based Human Action Recognition: A High-Definition Benchmark Dataset and Algorithms.** [19 August, 2024] [ArXiv, 2024]<br/>
  *Xiao Wang, Shiao Wang, Pengpeng Shao, Bo Jiang, Lin Zhu, Yonghong Tian.*<br/>
  [[Paper](https://arxiv.org/abs/2408.09764)] [[Code](https://github.com/Event-AHU/CeleX-HAR)]
* **OccMamba: Semantic Occupancy Prediction with State Space Models.** [19 August, 2024] [ArXiv, 2024]<br/>
  *Heng Li, Yuenan Hou, Xiaohan Xing, Xiao Sun, Yanyong Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.09859)]
* **Multi-Scale Representation Learning for Image Restoration with State-Space Model.** [19 August, 2024] [ArXiv, 2024]<br/>
  *Yuhong He, Long Peng, Qiaosi Yi, Chen Wu, Lu Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.10145)] 
* **MambaEVT: Event Stream based Visual Object Tracking using State Space Model.** [19 August, 2024] [ArXiv, 2024]<br/>
  *Xiao Wang, Chao wang, Shiao Wang, Xixi Wang, Zhicheng Zhao, Lin Zhu, Bo Jiang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.10487)] [[Code](https://github.com/Event-AHU/MambaEVT)]
* **Event Stream based Sign Language Translation: A High-Definition Benchmark Dataset and A New Algorithm.** [19 August, 2024] [ArXiv, 2024]<br/>
  *Xiao Wang, Yao Rong, Fuling Wang, Jianing Li, Lin Zhu, Bo Jiang, Yaowei Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.10488)] [[Code](https://github.com/Event-AHU/OpenESL)]
* **MUSE: Mamba is Efficient Multi-scale Learner for Text-video Retrieval.** [20 August, 2024] [ArXiv, 2024]<br/>
  *Haoran Tang, Meng Cao, Jinfa Huang, Ruyang Liu, Peng Jin, Ge Li, Xiaodan Liang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.10575)] [[Code](https://github.com/hrtang22/MUSE)]
* **MV-MOS: Multi-View Feature Fusion for 3D Moving Object Segmentation.** [20 August, 2024] [ArXiv, 2024]<br/>
  *Jintao Cheng, Xingming Chen, Jinxin Liang, Xiaoyu Tang, Xieyuanli Chen, Dachuan Li.*<br/>
  [[Paper](https://arxiv.org/abs/2408.10602)] [[Code](https://github.com/Chengjt1999/MV-MOS)]
* **OMEGA: Efficient Occlusion-Aware Navigation for Air-Ground Robot in Dynamic Environments via State Space Model.** [20 August, 2024] [ArXiv, 2024]<br/>
  *Junming Wang, Dong Huang, Xiuxian Guan, Zekai Sun, Tianxiang Shen, Fangming Liu, Heming Cui.*<br/>
  [[Paper](https://arxiv.org/abs/2408.10618)] [[Homepage](https://jmwang0117.github.io/OMEGA/)] [[Code](https://github.com/jmwang0117/Occ-Mamba)] 
* **DemMamba: Alignment-free Raw Video Demoireing with Frequency-assisted Spatio-Temporal Mamba.** [20 August, 2024] [ArXiv, 2024]<br/>
  *Shuning Xu, Xina Liu, Binbin Song, Xiangyu Chen, Qiubo Chen, Jiantao Zhou.*<br/>
  [[Paper](https://arxiv.org/abs/2408.10679)]
* **MambaDS: Near-Surface Meteorological Field Downscaling with Topography Constrained Selective State Space Modeling.** [20 August, 2024] [ArXiv, 2024]<br/>
  *Zili Liu, Hao Chen, Lei Bai, Wenyuan Li, Wanli Ouyang, Zhengxia Zou, Zhenwei Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2408.10854)]
* **HMT-UNet: A hybird Mamba-Transformer Vision UNet for Medical Image Segmentation.** [20 August, 2024] [ArXiv, 2024]<br/>
  *Mingya Zhang, Limei Gu, Tingshen Ling, Xianping Tao.*<br/>
  [[Paper](https://arxiv.org/abs/2408.11289)] [[Code](https://github.com/simzhangbest/HMT-Unet)]
* **MambaOcc: Visual State Space Model for BEV-based Occupancy Prediction with Local Adaptive Reordering.** [21 August, 2024] [ArXiv, 2024]<br/>
  *Yonglin Tian, Songlin Bai, Zhiyao Luo, Yutong Wang, Yisheng Lv, Fei-Yue Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.11464)] [[Code](https://github.com/Hub-Tian/MambaOcc)]
* **UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images.** [26 August, 2024] [ArXiv, 2024]<br/>
  *Enze Zhu, Zhan Chen, Dingkai Wang, Hanru Shi, Xiaoxuan Liu, Lei Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.11545)] [[Code](https://github.com/EnzeZhu2001/UNetMamba)]
* **MambaCSR: Dual-Interleaved Scanning for Compressed Image Super-Resolution With SSMs.** [21 August, 2024] [ArXiv, 2024]<br/>
  *Yulin Ren, Xin Li, Mengxi Guo, Bingchen Li, Shijie Zhao, Zhibo Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2408.11758)] [[Code](https://github.com/renyulin-f/MambaCSR)]
* **Scalable Autoregressive Image Generation with Mamba.** [22 August, 2024] [ArXiv, 2024]<br/>
  *Haopeng Li, Jinyue Yang, Kexin Wang, Xuerui Qiu, Yuhong Chou, Xin Li, Guoqi Li.*<br/>
  [[Paper](https://arxiv.org/abs/2408.12245)] [[Code](https://github.com/hp-l33/AiM)]
* **Adapt CLIP as Aggregation Instructor for Image Dehazing.** [22 August, 2024] [ArXiv, 2024]<br/>
  *Xiaozhe Zhang, Fengying Xie, Haidong Ding, Linpeng Pan, Zhenwei Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2408.12317)]
* **O-Mamba: O-shape State-Space Model for Underwater Image Enhancement.** [22 August, 2024] [ArXiv, 2024]<br/>
  *Chenyu Dong, Chen Zhao, Weiling Cai, Bo Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.12816)] [[Code](https://github.com/chenydong/O-Mamba)]
* **MSVM-UNet: Multi-Scale Vision Mamba UNet for Medical Image Segmentation.** [25 August, 2024] [ArXiv, 2024]<br/>
  *Chaowei Chen, Li Yu, Shiquan Min, Shunfang Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.13735)] [[Code](https://github.com/gndlwch2w/msvm-unet)]
* **ShapeMamba-EM: Fine-Tuning Foundation Model with Local Shape Descriptors and Mamba Blocks for 3D EM Image Segmentation.** [26 August, 2024] [ArXiv, 2024]<br/>
  *Ruohua Shi, Qiufan Pang, Lei Ma, Lingyu Duan, Tiejun Huang, Tingting Jiang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.14114)]
* **LoG-VMamba: Local-Global Vision Mamba for Medical Image Segmentation.** [26 August, 2024] [ArXiv, 2024]<br/>
  *Trung Dinh Quoc Dang, Huy Hoang Nguyen, Aleksei Tiulpin.*<br/>
  [[Paper](https://arxiv.org/abs/2408.14415)] [[Code](https://github.com/Oulu-IMEDS/LoG-VMamba)]
* **ZeroMamba: Exploring Visual State Space Model for Zero-Shot Learning.** [27 August, 2024] [ArXiv, 2024]<br/>
  *Wenjin Hou, Dingjie Fu, Kun Li, Shiming Chen, Hehe Fan, Yi Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2408.14868)] [[Code](https://anonymous.4open.science/r/ZeroMamba)]
* **MTMamba++: Enhancing Multi-Task Dense Scene Understanding via Mamba-Based Decoders.** [27 August, 2024] [ArXiv, 2024]<br/>
  *Baijiong Lin, Weisen Jiang, Pengguang Chen, Shu Liu, Ying-Cong Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2408.15101)] [[Code](https://github.com/EnVision-Research/MTMamba)]****


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
- **Mamba or RWKV: Exploring High-Quality and High-Efficiency Segment Anything Model.** [27 June, 2024] [ArXiv, 2024]<br/>
  *Haobo Yuan, Xiangtai Li, Lu Qi, Tao Zhang, Ming-Hsuan Yang, Shuicheng Yan, Chen Change Loy.*<br/>
  [[Paper](https://arxiv.org/abs/2406.19369)] [[Code](https://github.com/HarborYuan/ovsam)]
- **Scalable Visual State Space Model with Fractal Scanning.** [26 May, 2024] [ArXiv, 2024]<br/>
  *Lv Tang, HaoKe Xiao, Peng-Tao Jiang, Hao Zhang, Jinwei Chen, Bo Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.14480)]
- **MTMamba: Enhancing Multi-Task Dense Scene Understanding by Mamba-Based Decoders.** [14 July, 2024] [ArXiv, 2024]<br/>
  *Baijiong Lin, Weisen Jiang, Pengguang Chen, Yu Zhang, Shu Liu, Ying-Cong Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2407.02228)] [[Code](https://github.com/EnVision-Research/MTMamba)]
- **Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model for Few-Shot Class-Incremental Learning.** [8 July, 2024] [ArXiv, 2024]<br/>
  *Xiaojie Li, Yibo Yang, Jianlong Wu, Bernard Ghanem, Liqiang Nie, Min Zhang.*<br.>
  [[Paper](https://arxiv.org/abs/2407.06136)] [[Code](https://github.com/xiaojieli0903/Mamba-FSCIL)]
- **MambaVision: A Hybrid Mamba-Transformer Vision Backbone.** [10 July, 2024] [ArXiv, 2024]<br/>
  *Ali Hatamizadeh, Jan Kautz.*<br/>
  [[Paper](https://arxiv.org/abs/2407.08083)] [[Code](https://github.com/NVlabs/MambaVision)]
- **GroupMamba: Parameter-Efficient and Accurate Group Visual State Space Model.** [18 July, 2024] [ArXiv, 2024]<br/>
  *Abdelrahman Shaker, Syed Talal Wasim, Salman Khan, Juergen Gall, Fahad Shahbaz Khan.*<br/>
  [[Paper](https://arxiv.org/abs/2407.13772)] [[Code](https://github.com/Amshaker/GroupMamba)]

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
- **VideoMambaPro: A Leap Forward for Mamba in Video Understanding.** [27 June, 2024] [ArXiv, 2024]<br/>
  *Hui Lu, Albert Ali Salah, Ronald Poppe.*<br/>
  [[Paper](https://arxiv.org/abs/2406.19006)] [[Code](https://github.com/hotfinda/VideoMambaPro)]
- **DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark.** [30 May, 2024] [ArXiv, 2024]<br/>
  *Haoxing Chen, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Yaohui Li, Jun Lan, Huijia Zhu, Jianfu Zhang, Weiqiang Wang, Huaxiong Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.19707)] [[Code](https://github.com/chenhaoxing/DeMamba)]
- **QueryMamba: A Mamba-Based Encoder-Decoder Architecture with a Statistical Verb-Noun Interaction Module for Video Action Forecasting.** [4 July, 2024] [ArXiv, 2024]<br/>
  *Zeyun Zhong, Manuel Martin, Frederik Diederichs, Juergen Beyerer.*<br/>
  [[Paper](https://arxiv.org/abs/2407.04184)]
- **VideoMamba: Spatio-Temporal Selective State Space Model.** [11 July, 2024] [ArXiv, 2024]<br/>
  *Jinyoung Park, Hee-Seon Kim, Kangwook Ko, Minbeom Kim, Changick Kim.*<br/>
  [[Paper](https://arxiv.org/abs/2407.08476)] [[Code](https://github.com/jinyjelly/VideoMamba)]
- **Harnessing Temporal Causality for Advanced Temporal Action Detection.** [25 July, 2024] [ArXiv, 2024]<br/>
  *Shuming Liu, Lin Sui, Chen-Lin Zhang, Fangzhou Mu, Chen Zhao, Bernard Ghanem.*<br/>
  [[Paper](https://arxiv.org/abs/2407.17792)] [[Code](https://github.com/sming256/OpenTAD)]

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
- **FER-YOLO-Mamba: Facial Expression Detection and Classification Based on Selective State Space.** [9 May, 2024] [ArXiv, 2024]<br/>
  *Hui Ma, Sen Lei, Turgay Celik, Heng-Chao Li.*<br/>
  [[Paper](https://arxiv.org/abs/2405.01828)] [[Code](https://github.com/SwjtuMa/FER-YOLO-Mamba)]
- **OverlapMamba: Novel Shift State Space Model for LiDAR-based Place Recognition.** [13 May, 2024] [ArXiv, 2024]<br/>
  *Qiuchi Xiang, Jintao Cheng, Jiehao Luo, Jin Wu, Rui Fan, Xieyuanli Chen, Xiaoyu Tang.*<br/>
  [[Paper](https://arxiv.org/abs/2405.07966)]
- **TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction.** [27 May, 2024] [ArXiv, 2024]<br/>
  *Yinda Chen, Haoyuan Shi, Xiaoyu Liu, Te Shi, Ruobing Zhang, Dong Liu, Zhiwei Xiong, Feng Wu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.16847)] [[Code](https://github.com/ydchen0806/TokenUnify)]
- **MambaDepth: Enhancing Long-range Dependency for Self-Supervised Fine-Structured Monocular Depth Estimation.** [6 June, 2024] [ArXiv, 2024]<br/>
  *Ionuţ Grigore, Călin-Adrian Popa.*<br/>
  [[Paper](https://arxiv.org/abs/2406.04532)] [[Code](https://github.com/ionut-grigore99/MambaDepth)]
- **Q-Mamba: On First Exploration of Vision Mamba for Image Quality Assessment.** [13 June, 2024] [ArXiv, 2024]<br/>
  *Fengbin Guan, Xin Li, Zihao Yu, Yiting Lu, Zhibo Chen.*<br/>
  [[Paper](https://arxiv.org/abs/2406.09546)]
- **SUM: Saliency Unification through Mamba for Visual Attention Modeling.** [25 June, 2024] [ArXiv, 2024]<br/>
  *Alireza Hosseini, Amirhossein Kazerouni, Saeed Akhavan, Michael Brudno, Babak Taati.*<br/>
  [[Paper](https://arxiv.org/abs/2406.17815)] [[Code](https://github.com/Arhosseini77/SUM)]
- **VMambaCC: A Visual State Space Model for Crowd Counting.** [6 May, 2024] [ArXiv, 2024]<br/>
  *Hao-Yuan Ma, Li Zhang, Shuai Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2405.03978)]
- **MonoMM: A Multi-scale Mamba-Enhanced Network for Real-time Monocular 3D Object Detection.** [1 August, 2024] [ArXiv, 2024]<br/>
  *Youjia Fu, Zihao Xu, Junsong Fu, Huixia Xue, Shuqiu Tan, Lei Li.*<br/>
  [[Paper](https://arxiv.org/abs/2408.00438)]

### 2 Low-level Vision

#### 2.1 Image Denoising and Enhancement

- **U-shaped Vision Mamba for Single Image Dehazing.** [15 February, 2024] [ArXiv, 2024]<br/>
  *Zhuoran Zheng, Chen Wu.*<br/>[[Paper](https://arxiv.org/abs/2402.04139)] [[Code](https://github.com/zzr-idam)]
- **FreqMamba: Viewing Mamba from a Frequency Perspective for Image Deraining.** [15 April, 2024] [ArXiv, 2024]<br/>
  *Zou Zhen, Yu Hu, Zhao Feng.*<br/>[[Paper](https://arxiv.org/abs/2404.09476)]
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
- **LFMamba: Light Field Image Super-Resolution with State Space Model.** [18 June, 2024] [ArXiv, 2024]<br/>
  *Wang xia, Yao Lu, Shunzhou Wang, Ziqi Wang, Peiqi Xia, Tianfei Zhou.*<br/>[[Paper](https://arxiv.org/abs/2406.12463)]
- **Mamba-based Light Field Super-Resolution with Efficient Subspace Scanning.** [23 June, 2024] [ArXiv, 2024]<br/>
  *Ruisheng Gao, Zeyu Xiao, Zhiwei Xiong.*<br/>
  [[Paper](https://arxiv.org/abs/2406.16083)]
- **MxT: Mamba x Transformer for Image Inpainting.** [26 July, 2024] [ArXiv, 2024]<br/>
  *Shuang Chen, Amir Atapour-Abarghouei, Haozheng Zhang, Hubert P. H. Shum.*<br/>
  [[Paper](https://arxiv.org/abs/2407.16126)]
- **GMSR:Gradient-Guided Mamba for Spectral Reconstruction from RGB Images.** [13 May, 2024] [ArXiv, 2024]<br/>
  *Xinying Wang, Zhixiong Huang, Sifan Zhang, Jiawen Zhu, Lin Feng.*<br/>
  [[Paper](https://arxiv.org/abs/2405.07777)] [[Code](https://github.com/wxy11-27/GMSR)]
- **Dual Hyperspectral Mamba for Efficient Spectral Compressive Imaging.** [1 June, 2024] [ArXiv, 2024]<br/>
  *Jiahua Dong, Hui Yin, Hongliu Li, Wenbo Li, Yulun Zhang, Salman Khan, Fahad Shahbaz Khan.*<br/>
  [[Paper](https://arxiv.org/abs/2406.00449)] [[Code](https://github.com/JiahuaDong/DHM)]
- **HTD-Mamba: Efficient Hyperspectral Target Detection with Pyramid State Space Model.** [17 July, 2024] [ArXiv, 2024]<br/>
  *Dunbin Shen, Xuanbing Zhu, Jiacheng Tian, Jianjun Liu, Zhenrong Du, Hongyu Wang, Xiaorui Ma.*<br/>
  [[Paper](https://arxiv.org/abs/2407.06841)] [[Code](https://github.com/shendb2022/HTD-Mamba)]
- **Empowering Snapshot Compressive Imaging: Spatial-Spectral State Space Model with Across-Scanning and Local Enhancement.** [1 August, 2024] [ArXiv, 2024]<br/>
  *Wenzhe Tian, Haijin Zeng, Yin-Ping Zhao, Yongyong Chen, Zhen Wang, Xuelong Li.*<br/>
  [[Paper](https://arxiv.org/abs/2408.00629)]

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
- **Mamba24/8D: Enhancing Global Interaction in Point Clouds via State Space Model.** [25 June, 2024] [ArXiv, 2024]<br/>
  *Zhuoyuan Li, Yubo Ai, Jiahao Lu, ChuXin Wang, Jiacheng Deng, Hanzhi Chang, Yanzhe Liang, Wenfei Yang, Shifeng Zhang, Tianzhu Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.17442)]
- **Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection.** [18 June, 2024] [ArXiv, 2024]<br/>
  *Guowen Zhang, Lue Fan, Chenhang He, Zhen Lei, Zhaoxiang Zhang, Lei Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.10700)] [[Code](https://github.com/gwenzhang/Voxel-Mamba)]
- **Serialized Point Mamba: A Serialized Point Cloud Mamba Segmentation Model.** [17 July, 2024] [ArXiv, 2024]<br/>
  *Tao Wang, Wei Wen, Jingzhi Zhai, Kang Xu, Haoming Luo.*<br/>
  [[Paper](https://arxiv.org/abs/2407.12319)]

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
- **Dimba: Transformer-Mamba Diffusion Models.** [3 June, 2024] [ArXiv, 2024]<br/>
  *Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, Junshi Huang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.01159)] [[Homepage](https://dimba-project.github.io/)] [[Code](https://github.com/feizc/Dimba)]
- **Hamba: Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba.** [12 July, 2024] [ArXiv, 2024]<br/>
  *Haoye Dong, Aviral Chharia, Wenbo Gou, Francisco Vicente Carrasco, Fernando De la Torre.*<br/>
  [[Paper](https://arxiv.org/abs/2407.09646)] [[Homepage](https://humansensinglab.github.io/Hamba/)] [[Code](https://github.com/humansensinglab/Hamba)] 
- **InfiniMotion: Mamba Boosts Memory in Transformer for Arbitrary Long Motion Generation.** [13 July, 2024] [ArXiv, 2024]<br/>
  *Zeyu Zhang, Akide Liu, Qi Chen, Feng Chen, Ian Reid, Richard Hartley, Bohan Zhuang, Hao Tang.*<br/>
  [[Paper](https://arxiv.org/abs/2407.10061)] [[Homepage](https://steve-zeyu-zhang.github.io/InfiniMotion/)]
- **OPa-Ma: Text Guided Mamba for 360-degree Image Out-painting.** [15 July, 2024] [ArXiv, 2024]<br/>
  *Penglei Gao, Kai Yao, Tiandi Ye, Steven Wang, Yuan Yao, Xiaofeng Wang.*<br/>
  [[Paper](https://arxiv.org/abs/2407.10923)]


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
- **An Empirical Study of Mamba-based Pedestrian Attribute Recognition.** [14 July, 2024] [ArXiv, 2024]<br/>
  *Xiao Wang, Weizhe Kong, Jiandong Jin, Shiao Wang, Ruichong Gao, Qingchuan Ma, Chenglong Li, Jin Tang.*<br/>
  [[Paper](https://arxiv.org/abs/2407.10374)] [[Code](https://github.com/Event-AHU/OpenPAR)]

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
- **SOAR: Advancements in Small Body Object Detection for Aerial Imagery Using State Space Models and Programmable Gradients.** [5 May, 2024] [ArXiv, 2024]<br/>*Tushar Verma, Jyotsna Singh, Yash Bhartari, Rishi Jarwal, Suraj Singh, Shubhkarman Singh.*<br/>[[Paper](https://arxiv.org/abs/2405.01699)] [[Code](https://github.com/yash2629/S.O.A.R)]
- **GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification.** [11 July, 2024] [ArXiv, 2024]<br/>
  *Aitao Yang, Min Li, Yao Ding, Leyuan Fang, Yaoming Cai, Yujie He.*<br/>
  [[Paper](https://arxiv.org/abs/2407.08255)] [[Code](https://github.com/ahappyyang/GraphMamba)]

#### 1.3 Remote Sensing Image Change Detection

- **ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model.** [14 April, 2024] [ArXiv, 2024]<br/>
  *Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, Naoto Yokoya.*<br/>
  [[Paper](https://arxiv.org/abs/2404.03425)] [[Code](https://github.com/ChenHongruixuan/MambaCD)]
- **RSCaMa: Remote Sensing Image Change Captioning with State Space Model.** [2 May, 2024] [ArXiv, 2024]<br/>
  *Chenyang Liu, Keyan Chen, Bowen Chen, Haotian Zhang, Zhengxia Zou, Zhenwei Shi.*<br/>[[Paper](https://arxiv.org/abs/2404.18895)] [[Code](https://github.com/Chen-Yang-Liu/RSCaMa)]
- **CDMamba: Remote Sensing Image Change Detection with Mamba.** [6 June, 2024] [ArXiv, 2024]<br/>
  *Haotian Zhang, Keyan Chen, Chenyang Liu, Hao Chen, Zhengxia Zou, Zhenwei Shi.*<br/>
  [[Paper](https://arxiv.org/abs/2406.04207)] [[Code](https://github.com/zmoka-zht/CDMamba)]
- **A Mamba-based Siamese Network for Remote Sensing Change Detection.** [8 July, 2024] [ArXiv, 2024]<br/>
  *Jay N. Paranjape, Celso de Melo, Vishal M. Patel.*<br/>
  [[Paper](https://arxiv.org/abs/2407.06839)] [[Code](https://github.com/JayParanjape/M-CD)]

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
- **DMM: Disparity-guided Multispectral Mamba for Oriented Object Detection in Remote Sensing.** [10 July, 2024] [ArXiv, 2024]<br/>
  *Minghang Zhou, Tianyu Li, Chaofan Qiao, Dongyu Xie, Guoqing Wang, Ningjuan Ruan, Lin Mei, Yang Yang.*<br/>
  [[Paper](https://arxiv.org/abs/2407.08132)] [[Code](https://github.com/Another-0/DMM)]

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
- **HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation.** [11 May, 2024] [ArXiv, 2024]<br/>
  *Jiashu Xu.*<br/>
  [[Paper](https://arxiv.org/abs/2405.05007)]
- **SliceMamba for Medical Image Segmentation.** [11 July, 2024] [ArXiv, 2024]<br/>
  *Chao Fan, Hongyuan Yu, Luo Wang, Yan Huang, Liang Wang, Xibin Jia.*<br/>
  [[Paper](https://arxiv.org/abs/2407.08481)]

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
- **GFE-Mamba: Mamba-based AD Multi-modal Progression Assessment via Generative Feature Extraction from MCI.** [22 July, 2024] [ArXiv, 2024]<br/>
  *Zhaojie Fang, Shenghao Zhu, Yifei Chen, Binfeng Zou, Fan Jia, Linwei Qiu, Chang Liu, Yiyu Huang, Xiang Feng, Feiwei Qin, Changmiao Wang, Yeru Wang, Jin Fan, Changbiao Chu, Wan-Zhen Wu, Hu Zhao.*<br/>
  [[Paper](https://arxiv.org/abs/2407.15719)] [[Code](https://github.com/Tinysqua/GFE-Mamba)]

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
- **MMR-Mamba: Multi-Modal MRI Reconstruction with Mamba and Spatial-Frequency Information Fusion.** [27 June, 2024] [ArXiv, 2024]<br/>
  *Jing Zou, Lanqing Liu, Qi Chen, Shujun Wang, Xiaohan Xing, Jing Qin.*<br/>
  [[Paper](https://arxiv.org/abs/2406.18950)]
- **Deform-Mamba Network for MRI Super-Resolution.** [8 July, 2024] [ArXiv, 2024]<br/>
  *Zexin Ji, Beiji Zou, Xiaoyan Kui, Pierre Vera, Su Ruan.*<br/>
  [[Paper](https://arxiv.org/abs/2407.05969)]

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

- **Soft Masked Mamba Diffusion Model for CT to MRI Conversion.** [22 June, 2024] [ArXiv, 2024]<br/>
  *Zhenbin Wang, Lei Zhang, Lituan Wang, Zhenwei Zhang.*<br/>
  [[Paper](https://arxiv.org/abs/2406.15910)] [[Code](https://github.com/wongzbb/DiffMa-Diffusion-Mamba)]

- **SR-Mamba: Effective Surgical Phase Recognition with State Space Model.** [11 July, 2024] [ArXiv, 2024]<br/>
  *Rui Cao, Jiangliu Wang, Yun-Hui Liu.*<br/>
  [[Paper](https://arxiv.org/abs/2407.08333)] [[Code](https://github.com/rcao-hk/SR-Mamba)]

  

## Other Domains

coming soon
