# DemoSemanticSegmentationADAS

This repository provides a **simple baseline Semantic Segmentation demo** used in the UniPR ADAS course. It demonstrates how to train and test semantic segmentation models using the CityScapes dataset.

---

## Table of Contents

* [Setup](#setup)
* [Dataset](#dataset)
* [Folder Structure](#folder-structure)
* [Running the Demo](#running-the-demo)
* [Configuration](#configuration)

---

## Setup

1. **Clone the repository** into your projects directory:

```bash
git clone https://github.com/MarcelloCeresini/DemoSemanticSegmentationADAS.git
cd DemoSemanticSegmentationADAS
```

2. **Create a `data` directory** inside the project folder:

```bash
mkdir data
```

3. **Set up a Python virtual environment** and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If you want GPU acceleration, install PyTorch with CUDA support following the instructions [here](https://pytorch.org/get-started/locally/).

---

## Dataset

This project uses the [CityScapes dataset](https://www.cityscapes-dataset.com/downloads/).

1. Create an account with your university email.
2. Download the following splits:

   * `gtFine_trainvaltest`
   * `leftImg8bit_trainvaltest`

---

## Folder Structure

After downloading and extracting the dataset, organize your files as follows:

```
DemoSemanticSegmentationADAS/
├── main.py
├── requirements.txt
└── data/
    ├── gtFine_trainvaltest/
    │   └── gtFine/
    │       ├── train/
    │       ├── val/
    │       └── test/
    └── leftImg8bit_trainvaltest/
        └── leftImg8bit/
            ├── train/
            ├── val/
            └── test/
```

---

## Running the Demo

Launch the demo using:

```bash
python main.py
```

---

## Configuration

You can modify parameters directly in `main.py` to experiment with different settings:

* Batch size
* Model architecture
* Number of epochs
* Learning rate

This allows you to observe how changes affect training and evaluation results.

---

**Remember:** Always activate the virtual environment before running the demo to ensure correct dependencies are used.
