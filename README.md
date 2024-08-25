# COVID-19 Segmentation Using Deep Learning

## Introduction

The COVID-19 pandemic has posed a significant challenge to global healthcare systems. Medical imaging, particularly computed tomography (CT) scans, plays a vital role in the diagnosis and management of COVID-19. Accurate segmentation of infected regions in lung CT scans can assist radiologists in assessing the severity of the infection and planning treatment. This study leverages deep learning techniques to automate the segmentation process, aiming to improve accuracy and efficiency.

## Dataset

The dataset utilized in this study consists of annotated lung CT scans from COVID-19 patients. The scans are accompanied by ground truth masks indicating the infected regions. The dataset is publicly available and can be accessed from [zenodo dataset](https://zenodo.org/records/3757476).

## Methodology

### Data Preprocessing

The CT scans undergo several preprocessing steps, including normalization, resizing, and augmentation. These steps are crucial to ensure the model generalizes well to different variations in the input data.

### Model Architecture

We employ a modified U-Net++ architecture for segmentation. The U-Net++ model consists of an encoder-decoder structure with skip connections, allowing for efficient localization and precise segmentation. The encoder captures context through successive downsampling, while the decoder reconstructs the segmentation map through upsampling.


### Training

The model is trained using a combination of binary cross-entropy and Dice loss, which helps in handling class imbalance and improving segmentation accuracy. The training process is carried out using the Adam optimizer with a learning rate of 1e-4. The model is trained for 100 epochs with a batch size of 2.

```sh
## Implementation

### training

python trainval.py --encoder densenet201 --base unetplus --datadir /dataset --savedir_base ./figures_no_augment 

### Testing

python test.py --datadir /dataset --savedir_base /figures_no_augment -ei unetplus_densenet201


```sh
pip install -r requirements.txt
