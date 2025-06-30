# Stem Splitter 

A deep learning project leveraging the U-Net architecture to separate individual musical instruments from audio recordings. This project demonstrates a novel approach to music source separation using convolutional neural networks, with a focus on clarity, reproducibility, and practical results.

---

## Table of Contents
- [Motivation & Innovation](#motivation--innovation)
- [Dataset](#dataset)
- [Dependencies & Installation](#dependencies--installation)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results & Evaluation](#results--evaluation)
- [Future Work](#future-work)
- [Credits & Author](#credits--author)

---

## Motivation & Innovation

Traditional music source separation methods often rely on frequency masking after Fourier Transform analysis. Inspired by the success of U-Nets in image segmentation, this project explores their application to audio, treating source separation as a segmentation problem in the time-frequency domain. My approach is among the first to apply U-Nets directly to raw audio for instrument separation, aiming for improved performance and generalization.

---

## Dataset

- **MUSDB18 Dataset:**
  - 150 professionally mixed songs in Native Instruments stems format (.mp4)
  - Each song contains 5 stereo streams: mixture, drums, bass, accompaniment, vocals
  - 44.1kHz, AAC @256kbps
  - [More info on MUSDB18](https://sigsep.github.io/datasets/musdb.html)

*Note: Due to computational constraints, only 2 tracks were used for training in this project.*

---

## Dependencies & Installation

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- IPython

Install dependencies with:
```bash
pip install tensorflow keras numpy pandas matplotlib ipython
```

---

## Preprocessing

- Stereo audio is converted to mono for simplicity and efficiency.
- The left channel is used for training, the right for validation.
- Audio is chunked into segments of 64 time steps, paired with corresponding instrument tracks (e.g., drums).
- Data is stored in Pandas DataFrames for easy manipulation.

---

## Model Architecture

### U-Net for Audio

- **Encoder:** Stacked Conv1D layers with increasing filters (16 → 512), batch normalization, and LeakyReLU activations.
- **Decoder:** Conv1DTranspose layers for upsampling, skip connections via concatenation, and ReLU activations.
- **Regularization:** Dropout layers after upsampling stages.
- **Output:** A mask is generated and applied to the input audio to isolate the target instrument.

---

## Training Details

- **Loss Function:** Mean Absolute Error (MAE)
- **Optimizer:** Adam (learning rate = 1e-3)
- **Epochs:** 40
- **Batch Size:** (as per available memory)
- **Progress:** Training and validation loss are plotted for each epoch.

---

## Results & Evaluation

- The model was evaluated on a held-out test track.
- Both the original and separated tracks are provided for qualitative assessment.

---

## Future Work

- Train on the full MUSDB18 dataset for improved generalization.
- Expand the U-Net width to capture longer time dependencies.
- Explore stereo separation and multi-instrument output.
- Experiment with advanced loss functions and data augmentation.

---

## Credits & Author

Project by Ayush Sah  
[![GitHub: xARSENICx](https://img.shields.io/badge/GitHub-xARSENICx-blue?logo=github)](https://github.com/xARSENICx)



