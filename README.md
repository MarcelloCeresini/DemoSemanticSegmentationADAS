# DemoSemanticSegmentationADAS
This repo was created to share a simple baseline Semantic Segmentation demo showed in the UniPR ADAS course


## Setup
- Move into a directory where you have your coding projects
- Clone this repo with
  git clone https://github.com/MarcelloCeresini/DemoSemanticSegmentationADAS/
- Move inside the DemoSemanticSegmentationADAS folder that has been created
- Here, create a "data" directory
- Moreover, create a python virtual environment with
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
- If you want to install torch with CUDA, follow instructions at https://pytorch.org
- To download the dataset, go at https://www.cityscapes-dataset.com/downloads/, and create an account with your university email
- Download CityScapes dataset splits gtFine_trainvaltest and leftImg8bit_trainvaltest
- Create a folder structure like this:
  DemoSemanticSegmentationADAS
    main.py
    requirements.txt
    data/
      gtFine_trainvaltest/
        gtFine/
          train/
          val/
          test/
      leftImg8bit_trainvaltest/
        leftImg8bit/
          train/
          val/
          test/
  
