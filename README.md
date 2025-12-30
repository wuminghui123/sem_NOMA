# CSC-SA-Net: E2E Learning Massive MIMO for Multimodal Semantic Non-Orthogonal Transmission and Fusion

This is the official implementation of the paper:

**"E2E Learning Massive MIMO for Multimodal Semantic Non-Orthogonal Transmission and Fusion"**

Published in *IEEE Journal on Selected Areas in Communications (JSAC)*

📄 **Paper Link:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11299525/)

## Abstract

This paper investigates multimodal semantic non-orthogonal transmission and fusion in hybrid analog-digital massive multiple-input multiple-output (MIMO). A Transformer-based cross-modal source-channel semantic-aware network (CSC-SA-Net) framework is conceived, where channel state information (CSI) reference signal (RS), feedback, analog-beamforming/combining, and baseband semantic processing are data-driven end-to-end (E2E) optimized at the base station (BS) and user equipments (UEs).

CSC-SA-Net comprises five sub-networks:
- **BS-CSIRS-Net**: BS-side CSI-RS network
- **UE-CSANet**: UE-side channel semantic-aware network
- **BS-CSANet**: BS-side channel semantic-aware network
- **UE-MSFNet**: UE-side multimodal semantic fusion network
- **BS-MSFNet**: BS-side multimodal semantic fusion network

## System Architecture

<p align="center">
  <img src="fig/system.png" alt="System Architecture" width="800"/>
</p>

## Features

- End-to-end learning framework for massive MIMO semantic communication
- Non-orthogonal transmission and over-the-air fusion of multimodal semantics
- Joint optimization of CSI-RS design, feedback, beamforming, and semantic processing
- Three-stage task-driven training strategy
- Support for semantic segmentation downstream tasks

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+
- NumPy
- PIL (Pillow)
- tqdm
- matplotlib

Install dependencies:
```bash
pip install torch torchvision numpy pillow tqdm matplotlib
```

## Dataset

This project uses the **MFNet Dataset** (Multispectral Fusion Network Dataset) for semantic segmentation tasks with RGB and thermal infrared images.

🔗 **Dataset Source:** [http://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral](http://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral)

### Dataset Structure

After downloading, organize the dataset as follows:
```
ir_seg_dataset/
├── images/          # RGB and thermal images (4-channel)
├── labels/          # Segmentation labels
├── train.txt        # Training set file list
├── val.txt          # Validation set file list
└── test.txt         # Test set file list
```

## Project Structure

```
sem_NOMA/
├── main.ipynb       # Main notebook with complete implementation
├── my_log.py        # Logging utilities
├── README.md        # This file
└── util/
    ├── MF_dataset.py    # Dataset loader for MFNet
    ├── util.py          # Utility functions (accuracy, IoU calculation)
    └── augmentation.py  # Data augmentation functions
```

## Usage

### Training

Open `main.ipynb` in Jupyter Notebook or VS Code and run the cells sequentially. The notebook includes:

1. **Stage 1**: Pre-train UE/BS-MSFNet on the downstream semantic segmentation task
2. **Stage 2**: Jointly optimize BS-CSIRS-Net and UE/BS-CSANet for spectral efficiency
3. **Stage 3**: End-to-end training of the complete CSC-SA-Net framework

### Key Parameters

- `C_dim`: Semantic encoding channel dimension (default: 64)
- `n_class`: Number of segmentation classes (default: 9)
- `fc`: Carrier frequency (default: 28 GHz)
- SNR settings can be adjusted for different channel conditions

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@ARTICLE{Wu2025E2EMIMO,
  author={Wu, Minghui and Gao, Zhen},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={E2E Learning Massive MIMO for Multimodal Semantic Non-Orthogonal Transmission and Fusion}, 
  year={2025},
  doi={10.1109/JSAC.2025.3643817},
  keywords={Semantic communication; Image reconstruction; Massive MIMO; Array signal processing; Channel estimation; Symbols; Precoding; Antenna arrays; Receivers; Downlink; Massive MIMO; Deep learning; Semantic communication; Multimodal fusion; Non-orthogonal transmission}
}
```

## Keywords

`Massive MIMO` `Deep Learning` `Semantic Communication` `Multimodal Fusion` `Non-Orthogonal Transmission` `End-to-End Learning` `CSI Acquisition` `Beamforming`

## Authors

- **Minghui Wu** - Beijing Institute of Technology (BIT) - wuminghui@bit.edu.cn
- **Zhen Gao** (Corresponding Author) - Beijing Institute of Technology (BIT) - gaozhen16@bit.edu.cn

## Acknowledgments

This work was supported in part by:
- Natural Science Foundation of China (NSFC) under Grant 62471036 and Grant U2233216
- Shandong Province Natural Science Foundation under Grant ZR2025QA30 and Grant ZR2022YQ62
- Beijing Natural Science Foundation under Grant L242011

## License

This project is released for academic research use only. Please cite our paper if you use this code.
 
