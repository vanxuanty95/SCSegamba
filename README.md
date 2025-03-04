<div align="center">
  <h1>SCSegamba</h1>
</div>
<p align="center">
    <img src="./figures/LOGO.png" alt="LOGO" width="185" height="200" />
</p>


<div align="center">
<h4>[CVPR2025] SCSegamba: Lightweight Structure-Aware Vision Mamba for Crack Segmentation in Structures</h4>
</div>

<div align="center">
<h6>🌟 If this work is useful to you, please give this repository a Star! 🌟</h6>
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2503.01113"><img src="https://img.shields.io/badge/Arxiv-2503.01113-b31b1b?logo=arXiv" alt="arXiv" style="height:20px;"></a>
  <a href="https://www.apache.org/licenses/" style="margin-left:10px;"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License" style="height:20px;"></a>
</div>


## 📬News

- **2025-03-04**: The code for **SCSegamba** is publicly available in this repository📦!
- **2025-03-04**: The preprint of **SCSegamba** has been posted on [**📤️arXiv**](https://arxiv.org/abs/2503.01113)!
- **2025-03-01**: In the next few days, we will make some minor revisions and then publish the preprint on arXiv. The code will also be released shortly after the paper is published! **Stay tuned🥰**!
- **2025-02-27**: 🎉🎉🎉 We are thrilled to announce that our **SCSegamba** has been accepted to **CVPR 2025**! 

## 🕹Getting Started

#### Environment Setup

You can create your own conda environment for SCSegamba based on the following command⚙️:

```shell
conda create -n SCSegamba python=3.10 -y
conda activate SCSegamba
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv-full
pip install mamba-ssm==1.2.0
pip install timm lmdb mmengine numpy
```

#### Run

You can modify the parameters in the **main.py** file and run it with the following command:

``````shell
python main.py
``````

You can also use checkpoints for inference with the following command:

```shell
python test.py
```

Use the following commands to calculate the ODS, OIS, P, R, F1, mIoU metrics (You can find the SCSegamba test results on the TUT dataset in the `./results/results_test/TUT_results/` path and calculate the metrics using the following command.):

```shell
python eval_compute.py
cd eval
python evaluate.py
```

You can also follow the steps below to validate the results of our experiments on the TUT dataset

- **Download Checkpoint**: Get the [checkpoint](https://drive.google.com/file/d/1r36WUaCoeNjtfZN9BRS-uPMcRglQTGx3/view?usp=sharing) file we pre-trained on the [TUT](https://github.com/Karl1109/TUT) dataset

- **File Placement**: Move the downloaded checkpoint file to the designated path: `./checkpoints/weights/checkpoint_TUT/`

- **Run**: Change the relevant path in test.py and run this command: `python test.py`.

## 🤝 Citation

Please cite our work if it is useful for your research.

```
@misc{liu2025scsegamba,
      title={SCSegamba: Lightweight Structure-Aware Vision Mamba for Crack Segmentation in Structures}, 
      author={Hui Liu and Chen Jia and Fan Shi and Xu Cheng and Shengyong Chen},
      year={2025},
      eprint={2503.01113},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.01113}, 
}
```

## 🗓️ TODO

- [🟢 Complete] **arXiv Preprint Release**  
- [🟢 Complete] **Open source code at this repository**
- [🟡 In Progress] Add a description of the method in ReadMe
- [🟡 In Progress] Add visualization of experiment results in ReadMe

## 🏷️License

This project is released under the [**Apache 2.0**](https://www.apache.org/licenses/) license.

## 🫡Acknowledgment

This work stands on the shoulders of the following **open-source projects**:

<div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin: 20px 0;">
  <div>
    <a href="https://github.com/ChenhongyiYang/PlainMamba" target="_blank">PlainMamba</a> 
    <a href="https://arxiv.org/abs/2403.17695">[Paper]</a>
  </div>
  <div>
    <a href="https://github.com/yhlleo/DeepCrack" target="_blank">DeepCrack</a> 
    <a href="https://www.sciencedirect.com/science/article/abs/pii/S0925231219300566">[Paper]</a>
  </div>
  <div>
    <a href="https://github.com/open-mmlab/mmclassification" target="_blank">mmclassification</a>
  </div>
</div>

## 📟Contact

If you have any other questions, feel free to contact me at **liuhui1109@stud.tjut.edu.cn** or **liuhui@ieee.org**.
