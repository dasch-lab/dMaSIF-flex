# Enhancing Antibody–Antigen Interaction Prediction with Atomic Flexibility

---

## Abstract

Antibodies are indispensable components of the immune system, known for their specific binding to antigens. Beyond their natural immunological functions, they are fundamental in developing vaccines and therapeutic interventions for infectious diseases. The complex architecture of antibodies, particularly their variable regions responsible for antigen recognition, presents significant challenges for computational modeling.

Recent advancements in deep learning have markedly improved protein structure prediction; however, accurately modeling antibody–antigen (Ab–Ag) interactions remains challenging due to the inherent flexibility of antibodies and the dynamic nature of binding processes.  

In this study, we examine the use of predicted Local Distance Difference Test (pLDDT) scores as indicators of residue and side-chain flexibility to model Ab–Ag interactions through a fingerprint-based approach. We demonstrate the significance of flexibility in different antibody-specific tasks, enhancing the predictive accuracy of Ab–Ag interaction models by **4%**, resulting in an **AUC-ROC of 92%**. Additionally, we achieve state-of-the-art performance in paratope prediction.

These results emphasize the importance of accounting for conformational flexibility in modeling Ab–Ag interactions and show that pLDDT can effectively serve as a proxy for these dynamic features. By optimizing antibody flexibility using pLDDT, antibodies can be engineered to improve affinity or breadth for a specific target—particularly beneficial against highly variable pathogens such as HIV and SARS-CoV-2.

---

## Repository Structure

- dataset/ # (empty) Download dataset from Zenodo (link below)
- dataset_list/ # Dataset names for site and search prediction (PPI & Ab–Ag)
- model/ # Pretrained dMaSIF-flex models for Ab–Ag tasks
- src/ # Modified dMaSIF code
- Arguments.py # Script arguments for training & inference
- environment.txt # Conda/virtualenv environment specification
- main_inference.py # Inference script
- main_training.py # Training script


---

## Dataset Download

The `dataset/` folder is initially empty.  
You must download the dataset from Zenodo and place it inside `dataset/`:

🔗 **[Zenodo Dataset Link](https://zenodo.org/records/16782978?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjBmMGNkNjAxLTU5M2YtNGZmYy05ODgwLWQ1MjBlNzkxNWE5YSIsImRhdGEiOnt9LCJyYW5kb20iOiI1ZWNiNmFhMDUxZmMwNGI1ZWI0OTA5YWU1YzRlMDUwOCJ9.PWpOFONn1fPNFi1JpBl8-bbMDmGgxI1IP-iLfYHxpKTgEHDXv1RVmkHUfhJPx5u3j6MposylmMEbAiRBiK7TJg)**

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/dMaSIF-flex.git
   cd dMaSIF-flex

2. **Create the environment:**
    ```bash
    conda create --name abflex --file environment.txt
    conda activate abflex

---
## Usage

1. **Training:**
   ```bash
   python main_training.py 


2. **Inference:** \
  model/ directory contains pretrained dMaSIF-flex models for:
  - Site prediction (Ab–Ag and PPI)
  - Search prediction (Ab–Ag and PPI)
  
    ```bash
    python main_inference.py

---
## Citation
```bibtex
@article{Joubbiflex2025,
  title={Enhancing Antibody–Antigen Interaction Prediction with Atomic Flexibility},
  author={Joubbi et al.},
  journal={PLOS Computational Biology},
  year={2025}
}