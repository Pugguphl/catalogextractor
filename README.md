# Archaeological Catalog Collection Method Based on Large Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2412.20088-b31b1b.svg)](https://arxiv.org/pdf/2412.20088)

This repository contains code for automatically extracting archaeological catalog information using computer vision and large language models. The method processes images of pottery and other archaeological artifacts to extract structured data for analysis and documentation.

## Overview

The archaeological catalog collection method leverages state-of-the-art vision-language models to:

- Detect and localize artifacts in catalog images
- Extract catalog metadata
- Generate structured output for archaeological research

For technical details, please refer to [our paper](https://arxiv.org/pdf/2412.20088).



**Installation: catalogextractor**
1. Clone the package repository and navigate to the pdfextractor directory.

```bash
git clone https://github.com/Pugguphl/catalogextractor.git
cd catalogextractor/
```
2. Create a new Anaconda environment named `pdfextractor` with Python 3.10.

```bash
conda create -n catalogextractor python=3.10 -y
```

3. Activate the newly created environment.

```bash
conda activate catalogextractor
```

4. Install the required dependencies.

```bash
pip install -r requirements.txt
```



**Installation: Grounding Dino**

1.Clone the GroundingDINO repository from GitHub.

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

2. Change the current directory to the GroundingDINO folder.

```bash
cd GroundingDINO/
```

3. Install the required dependencies in the current directory.

```bash
pip install -e .
```

4. Download pre-trained model weights.

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
5. If the download command in step 4 fails, download the model manually:
   
   Visit https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth directly in your browser
   
   Save the downloaded file to the `weights` folder you created in step 3


**Running the Pottery Information Extractor**

1. Navigate back to the catalogextractor root directory.

```bash
cd ..  # If you're in the GroundingDINO directory
```

2. Run the pottery_info_extractor.py script with the required arguments:

```bash
python pottery_info_extractor.py --output_dir /path/to/output/directory --image_dir /path/to/pottery/images/folder --config_gd /path/to/grounding_dino/config --pretrained_weight_gd /path/to/grounding_dino/weights
```

3. Replace the path arguments with your actual paths:
    - `--output_dir`: The directory where extracted information will be saved
    - `--image_dir`: The directory containing pottery catalog images to be processed
    - `--config_gd`: Path to the Grounding DINO model configuration file
    - `--pretrained_weight_gd`: Path to the Grounding DINO pre-trained model weights

4. Example usage:

```bash
python pottery_info_extractor.py --output_dir ./results --image_dir ./pottery_catalog_images --config_gd GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --pretrained_weight_gd GroundingDINO/weights/groundingdino_swint_ogc.pth
```
