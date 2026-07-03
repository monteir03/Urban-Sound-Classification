# Urban Sound Classification

Implementation of deep learning models for sound classification using TensorFlow/Keras on the UrbanSound8K dataset.

## Overview

This project implements and compares multiple deep learning architectures for classifying urban sounds into 10 distinct categories. The models are trained and evaluated on the UrbanSound8K dataset, which contains 8,732 audio clips sampled at 22kHz from 10 different urban sound classes.

## Project Structure

The repository contains the following Jupyter notebooks:

### 1. **features.ipynb**
   - Feature extraction and preprocessing pipeline
   - Generates MFCC (Mel-Frequency Cepstral Coefficients) features from raw audio files
   - Handles audio augmentation techniques to balance the dataset
   - Creates feature vectors for model training

### 2. **CNN_DataImbalance.ipynb**
   - Convolutional Neural Network (CNN) trained on the original, imbalanced dataset
   - Serves as baseline model without data augmentation
   - Demonstrates the impact of class imbalance on model performance

### 3. **CNN_DataAugmented.ipynb**
   - CNN model trained on augmented and balanced dataset
   - Applies data augmentation techniques to improve generalization
   - Shows performance improvements through balanced training data

### 4. **LSTM_DataImbalance.ipynb**
   - Long Short-Term Memory (LSTM) network on imbalanced data
   - Captures temporal dependencies in audio features
   - Baseline LSTM performance for comparison

### 5. **LSTM_DataAugmented.ipynb**
   - LSTM model trained on augmented dataset
   - Combines temporal feature learning with balanced data
   - Demonstrates the effectiveness of augmentation on sequential models

## Dataset

**UrbanSound8K Dataset** contains 10 classes of urban sounds:
1. Air conditioner
2. Car horn
3. Children playing
4. Dog bark
5. Drilling
6. Engine idling
7. Gunshot
8. Jackhammer
9. Siren
10. Street music

The dataset is split into 10 folds for cross-validation, and features include metadata about fold, class ID, and salience level.

## Key Features

### Feature Extraction
- **MFCC (40 coefficients)**: Mel-frequency cepstral coefficients that represent audio in a perceptually meaningful way
- **Normalization**: Features are normalized for consistent model training
- **Data Augmentation**: Techniques applied to balance class distribution and improve model robustness

### Models Implemented

#### Convolutional Neural Network (CNN)
- Effective at capturing spatial patterns in spectrograms
- Uses convolutional layers to learn hierarchical feature representations
- Includes regularization (L2) to prevent overfitting

#### Long Short-Term Memory (LSTM)
- Captures temporal dependencies in audio sequences
- Processes MFCC features as time series data
- Better suited for sequential audio feature analysis

### Training Techniques
- **Data Augmentation**: Improves model generalization by artificially expanding the training set
- **K-Fold Cross-Validation**: Ensures robust evaluation across different data splits
- **Class Balancing**: Handles imbalanced dataset through augmentation
- **Regularization**: L2 regularization and dropout to prevent overfitting

## Results

The project compares model performance across:
- **With vs. Without Data Augmentation**: Shows significant improvement with balanced datasets
- **CNN vs. LSTM**: Different architectures' effectiveness on audio classification
- **Imbalanced vs. Balanced Data**: Impact of class distribution on model accuracy

Key metrics include:
- Accuracy
- Precision, Recall, F1-Score
- Confusion matrices for detailed class-wise analysis

## Dependencies

```
numpy
pandas
librosa
matplotlib
seaborn
scikit-learn
tensorflow
keras
```

## Usage

1. **Feature Extraction**: Run `features.ipynb` to generate MFCC features from the UrbanSound8K dataset
2. **Model Training**: Execute either CNN or LSTM notebooks to train models on:
   - Original imbalanced dataset
   - Augmented balanced dataset
3. **Evaluation**: Compare model performance across different configurations

## Workflow

```
Raw Audio Files (UrbanSound8K)
         ↓
   Feature Extraction (MFCC)
         ↓
   Data Augmentation & Balancing
         ↓
    Model Training (CNN/LSTM)
         ↓
    Model Evaluation & Comparison
```

## Key Insights

- **Data Augmentation Impact**: Models trained on augmented data show improved performance and better generalization
- **Architecture Comparison**: CNN and LSTM architectures capture different aspects of audio patterns
- **Class Imbalance**: Addressing class imbalance through augmentation is crucial for robust sound classification
- **Feature Representation**: MFCC features combined with deep learning effectively capture urban sound characteristics

## Future Improvements

- Experiment with additional feature extraction methods (Spectrograms, Log-Mel features)
- Implement ensemble methods combining CNN and LSTM
- Explore transfer learning approaches using pre-trained audio models
- Test attention mechanisms for better temporal feature weighting
- Implement real-time prediction pipeline

## References

- UrbanSound8K Dataset: https://urbansounddataset.weebly.com/urbansound8k.html
- Librosa Documentation: https://librosa.org/
- TensorFlow/Keras Documentation: https://www.tensorflow.org/

## License

This project is open source and available for educational and research purposes.

---

**Author**: monteir03  
**Created**: February 2024
