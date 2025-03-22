# Human Activity Recognition using CNN-LSTM

This project implements a deep learning model that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to recognize human activities from smartphone sensor data. The model is trained on the UCI HAR Dataset, which contains measurements from smartphone accelerometers and gyroscopes.

## Project Overview

The goal of this project is to classify human activities into six categories:
1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

## Dataset

The project uses the UCI Human Activity Recognition dataset, which contains data collected from 30 subjects performing activities of daily living while carrying a waist-mounted smartphone with embedded inertial sensors.

### Dataset Structure
- `UCI HAR Dataset/`: Root directory containing all data files
  - `train/`: Training data (70% of subjects)
  - `test/`: Test data (30% of subjects)
  - Each directory contains:
    - Inertial sensor data
    - Activity labels
    - Subject identifiers

## Model Architecture

The model combines CNN and LSTM layers to effectively capture both spatial and temporal patterns in the sensor data:

1. Input Layer: Accepts preprocessed sensor data
2. CNN Layers: Extract spatial features from the input
3. LSTM Layers: Process temporal dependencies
4. Dense Layers: Final classification

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
   ```
3. Download and extract the UCI HAR Dataset into the project directory

## Usage

Run the main script:
```bash
python lstm.py
```

The script will:
1. Load and preprocess the data
2. Train the CNN-LSTM model
3. Evaluate model performance
4. Generate visualization plots
5. Save the trained model

## Model Performance

The model achieves competitive performance on the HAR task:
- Training accuracy curves are saved as 'training_history.png'
- Final model weights are saved as 'har_cnn_lstm_model.h5'

## Training Visualization

The training process generates plots showing:
- Training vs. Validation Accuracy
- Training vs. Validation Loss

These visualizations help in monitoring the model's learning progress and identifying potential overfitting.

## Model Features

- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Batch processing for efficient training
- Data preprocessing and normalization
- One-hot encoded activity labels

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- UCI Machine Learning Repository for providing the HAR dataset
- The deep learning community for various insights and best practices
