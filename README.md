# Music Genre Classification

This Advanced Machine Learning project aims to classify music into one of 10 genres using Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and ensemble methods.

## Project Overview

This project implements and compares multiple deep learning architectures to classify audio tracks into 10 possible genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.

## Dataset


The dataset used for this project is the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), one of the most widely used public
datasets for evaluation in machine listening research for music genre recognition (MGR). The
files were collected in 2000-2001 from a variety of sources, including personal CDs, radio,
microphone recordings, in order to represent a variety of recording conditions.
The dataset holds the .wav files representing the recorded songs in the genres original
folder, and a folder images original, containing the MEL spectrograms that were used to
train the CNN model. The dataset also includes two .csv files containing features extracted
directly from the .wav file of each song. In particular, one file includes features computed
on the whole song (30 sec), while the other one contains the same features computed on 10
splits of the original song (each 3 seconds long).

An modified version of the dataset, which was used in this project, is available in the [GTZAN 2.0](https://www.kaggle.com/datasets/mattiaingrassia/gtzan-2-0) dataset, that contains 5 different versions already split in train/validation/test:

- Original: original images contained in the GTZAN dataset.
- Original Augmented: augmented images obtained starting from the original images.
- Cropped: cropped version of the original images.
- Cropped Augmented: augmented images obtained starting from the cropped version of the original images.
- Cropped Augmented & Noise Injection: noisy and augmented images obtained
starting from the cropped version of the original images.

``` bash
GTZAN 2.0
├── Augmented_GTZAN/
│   ├── original/
│   ├── cropped/
│   ├── cropped_aug/
│   ├── original_aug/
│   └── original_aug_inj/
├── features_3_sec.csv
├── features_30_sec.csv
├── RNN.keras              // Weights of the best RNN model
├── CNN.keras              // Weights of the best CNN model
├── indices_train.json
├── indices_test.json
└── indices_val.json
```

## Project Structure

### Notebooks

1. **`data_augmentation.ipynb`** - Data Augmentation Techniques
   - Spectrogram cropping strategies
   - Noise injection methods
   - Augmentation pipeline setup

2. **`cnn.ipynb`** - Convolutional Neural Network Models
   - **Model 0.0**: Baseline naive CNN on original images
   - **Model 1.0**: Improved CNN on original images
   - **Model 1.1**: Improved CNN on cropped images
   - **Model 1.2**: Improved CNN with data augmentation
   - **Model 1.3**: Improved CNN with augmentation + noise injection
   - **Model 1.4**: Improved CNN on augmented original images
   - Includes visualizations: loss/accuracy curves, confusion matrices, class-wise metrics

3. **`rnn.ipynb`** - Recurrent Neural Network Models
   - **Model 0.0**: Baseline LSTM on raw audio features
   - **Model 1.0**: Advanced LSTM architecture
   - **Model 1.1**: Advanced LSTM with normalized data
   - **Model 1.2**: Advanced LSTM with data augmentation + noise injection
   - Operates on temporal sequences (10 timesteps × 57 features)
   - Includes comprehensive evaluation metrics and comparisons

4. **`ensemble_models.ipynb`** - Ensemble Learning Methods
   - **Model 2.0**: Hybrid model combining CNN features + hand-crafted audio features
   - **Model 3.1**: Ensemble using probability sum
   - **Model 3.2**: Ensemble using voting with confidence
   - **Model 3.3**: Ensemble using probability multiplication
   - Combines predictions from CNN and RNN for improved accuracy

## Getting Started

### Prerequisites

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebooks (Kaggle)

1. Import the notebook on kaggle (clear the cells output if the file size is too big)
2. Add the GTZAN 2.0 dataset to the workspace
3. Update the `BASE_PATH` variables for Kaggle
4. Turn on the GPU (P100 is suggested)
5. Run the notebook

### Running the Notebooks (Other systems)

1. Connect the notebook to the environment containing all the necessary packages
2. Update the `BASE_PATH` variable in each notebook to match your dataset location
3. Run the notebook

## Model Comparison

Each notebook includes:

- Training and validation loss/accuracy plots
- Confusion matrices for error analysis
- Class-wise metrics (precision, recall, F1-score)
- Overall performance metrics
- Radar charts comparing multiple models

## Results

Models are evaluated on a held-out test set. Performance metrics include:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall

The ensemble models typically outperform individual architectures.
