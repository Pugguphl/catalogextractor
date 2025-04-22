**Installation: pdfextractor**
1. Clone the package repository and navigate to the pdfextractor directory.

```bash
git clone https://github.com/Pugguphl/pdfextractor.git
cd pdfextractor/
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


**Running the Pottery Information Extractor**

1. Run the pottery_info_extractor.py script with the required arguments:

```bash
python pottery_info_extractor.py -output_dir /path/to/output/directory -image_dir /path/to/pottery/images/folder -config_gd /path/to/grounding_dino/config -pretrained_weight_gd /path/to/grounding_dino/weights
```

2. Replace the path arguments with your actual paths:
    - `-output_dir`: The directory where extracted information will be saved
    - `-image_dir`: The directory containing pottery catalog images to be processed
    - `-config_gd`: Path to the Grounding DINO model configuration file
    - `-pretrained_weight_gd`: Path to the Grounding DINO pre-trained model weights

3. Example usage:

```bash
python pottery_info_extractor.py --output_dir ./results --image_dir ./pottery_catalog_images --config_gd GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --pretrained_weight_gd weights/groundingdino_swint_ogc.pth
```
