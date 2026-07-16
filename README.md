# Breast Density Classification: ConvNeXt vs Vision-Language Models

**Comparison of deep learning approaches for automated mammographic breast density assessment using BI-RADS classification**

---

## Overview

This repository contains the implementation of my master's thesis research comparing CNN-based and multimodal approaches for breast density classification in screening mammography. The study evaluates **ConvNeXt** and **BioMedCLIP** models across three learning scenarios:

- Zero-shot classification with BioMedCLIP  
- Linear probing with BioMedCLIP  
- End-to-end fine-tuning with ConvNeXt

## Key Findings

| Model | Learning Approach | Accuracy | F1-Score |
|-------|------------------|----------|----------|
| BioMedCLIP | Zero-shot | 0.47 | 0.31 |
| BioMedCLIP | Linear Probe | 0.57 | 0.50 |
| **ConvNeXt** | **Fine-tuned** | **0.69** | **0.66** |

The fine-tuned ConvNeXt model achieved the best performance, demonstrating that domain-specific fine-tuning remains crucial even when compared to advanced pre-trained multimodal models.

## Project Structure

```
tesisMamogra/
├── config.py               # Hyperparameters and paths (edit before running)
├── requirements.txt        # Python dependencies
│
├── data/
│   └── dataset.py          # ComplexMedicalDataset + MyDatamodule (shared by all models)
│
├── models/
│   ├── clip_probe.py       # BioMedCLIP linear probe (frozen encoder + trainable head)
│   ├── convnext_model.py   # ConvNeXt-Base fine-tuning
│   └── vgg16_model.py      # VGG16 fine-tuning
│
├── train_clip.py           # Run BioMedCLIP linear probing experiment
├── train_convnext.py       # Run ConvNeXt fine-tuning experiment
├── train_vgg16.py          # Run VGG16 fine-tuning experiment
│
└── notebooks/
    ├── extractDensities.ipynb   # Label extraction from radiology reports
    ├── zeroShotBiomed.ipynb     # Zero-shot BioMedCLIP experiments
    └── aux.ipynb                # Auxiliary visualization helpers
```

## Key Features

- **Multi-Class Classification:** Four BI-RADS breast density categories  
- **Transfer Learning:** ConvNeXt (ImageNet) and BioMedCLIP (PubMed)
- **Vision-Language Comparison:** CNN vs. multimodal approaches
- **Multiple Learning Paradigms:** Zero-shot, linear probing, fine-tuning
- **Medical-Aware Preprocessing:** Histogram matching and contrast enhancement
- **Comprehensive Evaluation:** Accuracy, F1-score, confusion matrices



## Dataset

**Source:** San José Hospital at TecSalud, Tecnológico de Monterrey, Monterrey, Mexico  
**Time Period:** 2014–2019  
**Total Cases:** 1,160 screening mammography exams  
**Images:** 4,640 mammographic images (MLO and CC views for both breasts)  
**Reports:** 1,160 radiology reports in Spanish (translated to English)

### BI-RADS Density Distribution (Balanced)
- **Heterogeneously dense:** ~450 images  
- **Scattered fibroglandular density:** ~450 images  
- **Extremely dense:** ~450 images  
- **Fatty predominance:** ~450 images  


### Dataset Structure
The dataset must be stored in a subdirectory within .data/ containing the images and JSON files with dataset information. The structure should follow this format:
```console
.data/
└── your_dataset_name/
    ├── 4kimages/
    │   ├── S0018466_2016_LMLO.tif
    │   ├── S0011765_2018_LCC.tif
    │   └── ...
    ├── train.json
    └── test.json 
```

### JSON File Format
Both train.json and test.json should contain metadata for each image:


```json
[
  {
    "filename": "S0018466_2016_LMLO.tif",
    "report": "Heterogeneously dense",
    "image_path": "4kimages/S0018466_2016_LMLO.tif"
  }
]
```


### Report-to-Label Mapping
The code automatically maps textual density descriptions to numeric labels:

| Report Text                                       | Label | BI-RADS Category              |
|---------------------------------------------------|-------|-------------------------------|
| Fatty predominance                                | 0     | Almost entirely fatty         |
| Characterized by scattered areas of pattern density | 1   | Scattered density             |
| Heterogeneously dense                             | 2     | Heterogeneously dense         |
| Extremely dense                                   | 3     | Extremely dense               |



### Data Split Behavior:

If only train.json exists: The code automatically splits it into training (85%) and validation (15%) sets
If both train.json and test.json exist: Train file is split for training/validation, test file is used for final evaluation



## Installation

Clone the repository:
```console
git clone https://github.com/yourusername/breast-density-classification.git
cd breast-density-classification
```


Create a virtual environment:
```console
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```


Install dependencies:
```console
pip install -r requirements.txt
```

## Usage

1. **Configure paths** — edit `config.py` and set `DATA_DIR` to the folder containing your `train.json`, `test.json`, and images.

2. **Run an experiment:**

```console
# BioMedCLIP linear probing
python train_clip.py

# ConvNeXt fine-tuning
python train_convnext.py

# VGG16 fine-tuning
python train_vgg16.py
```

3. **Monitor training** — logs are written to `logging_tests/` and viewable with TensorBoard:

```console
tensorboard --logdir=logging_tests
```

All three scripts follow the same pattern: they read hyperparameters from `config.py`, load data from `data/dataset.py`, and save TensorBoard logs under their respective names (`linear_probe/`, `convnext/`, `vgg16/`).

## 🔬 Methodology

### Preprocessing
- Cleaning and correcting Spanish radiology reports  
- Translation to English  
- Regex-based label extraction  
- BI-RADS density standardization  
- Histogram matching for contrast normalization  
- Class balancing (~450 images per class)

### Architectures

**ConvNeXt**
- End-to-end fine-tuning  
- Pretrained on ImageNet  
- AdamW optimization  

**BioMedCLIP**
- Pretrained on 15M PubMed Central image–text pairs  
- Zero-shot and linear probing experiments  
- Contrastive vision–language embeddings  

### Metrics
- Accuracy  
- Macro F1-score  
- Confusion matrices  

---

## 📈 Results Analysis

### Key Findings
- ConvNeXt fine-tuning achieved the strongest performance (0.69 accuracy).  
- Zero-shot BioMedCLIP struggled with subtle breast density differences.  
- Linear probing improved BioMedCLIP performance but still lagged behind ConvNeXt.  
- Adjacent BI-RADS density categories showed expected confusion patterns.  

---

## 💡 Key Insights

### Why ConvNeXt Performed Best
- Full gradient updates across the entire network  
- Vision-only specialization without text–image alignment gaps  
- Direct numeric label optimization  
- Strong adaptation to mammographic textures and density patterns  

### Vision–Language Limitations
- Generic BI-RADS prompts lack granular descriptive detail  
- Representation mismatch between textual and visual modalities  
- Frozen encoders restrict domain adaptation  
- Token-based classification struggles with fine-grained density cues  

---

## 🔮 Future Work
- Improved, more descriptive textual prompts for VLMs  
- Multimodal fine-tuning using both images and radiology reports  
- Larger and more diverse mammography datasets  
- Additional metrics (AUC-ROC, per-class precision/recall)  
- Testing alternative architectures and hybrid models  
- Evaluation in real clinical screening workflows  


## 🤝 Acknowledgments

- **TecSalud** for providing the mammography dataset  
- **Tecnológico de Monterrey** for institutional support  
- **CONAHCYT** (Grant #1317813)  
- **ELADAIS Project** funded by the Spanish Ministry of Economic Affairs  
- **Microsoft AI for Good Research Lab** for Azure sponsorship credits  

---

## 📝 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.


---

**Note:** This repository is part of a master's thesis research project. The dataset is not publicly available due to patient privacy regulations, but the code and methodology are provided for reproducibility and educational purposes.
