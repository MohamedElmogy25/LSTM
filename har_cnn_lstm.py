# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import os  # For file path operations
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For dataset splitting
from tensorflow.keras.models import Sequential  # For creating sequential neural network
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization  # Neural network layers
from tensorflow.keras.utils import to_categorical  # For one-hot encoding of labels
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Training callbacks
import matplotlib.pyplot as plt  # For plotting training results

# Function to load the HAR (Human Activity Recognition) dataset
def load_data(data_path):
    # Load features (accelerometer and gyroscope data)
    X_data = pd.read_csv(os.path.join(data_path, dataset, f'X_{dataset}.txt'), sep=r'\s+', header=None)
    # Load activity labels (WALKING, WALKING_UPSTAIRS, etc.)
    y_data = pd.read_csv(os.path.join(data_path, dataset, f'y_{dataset}.txt'), header=None)[0]
    # Load subject IDs for potential subject-wise analysis
    subject_data = pd.read_csv(os.path.join(data_path, dataset, f'subject_{dataset}.txt'), header=None)[0]
    return X_data, y_data, subject_data

# Function to preprocess the data for CNN-LSTM model
def preprocess_data(X_train, X_test):
    # Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape data for CNN-LSTM architecture
    # Format: (samples, timesteps, features)
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_timesteps = 1  # Each sample represents one timestep
    
    # Reshape training and test data to match the input shape required by the CNN-LSTM model
    X_train_reshaped = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
    X_test_reshaped = X_test_scaled.reshape(X_test.shape[0], n_timesteps, n_features)
    
    return X_train_reshaped, X_test_reshaped

# Function to create the CNN-LSTM model architecture
def create_model(input_shape, n_classes):
    model = Sequential([
        # CNN layers for feature extraction
        # First convolutional block
        Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),  # Normalize layer outputs
        MaxPooling1D(pool_size=2, padding='same'),  # Reduce spatial dimensions
        
        # Second convolutional block with increased filters
        Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding='same'),
        
        # LSTM layers for temporal feature learning
        # First LSTM layer with return sequences for stacking
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),  # Prevent overfitting
        
        # Second LSTM layer
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers for classification
        Dense(64, activation='relu'),  # Fully connected layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')  # Output layer for multi-class classification
    ])
    
    # Configure model training parameters
    model.compile(optimizer='adam',  # Adaptive learning rate optimization
                  loss='categorical_crossentropy',  # Suitable for multi-class classification
                  metrics=['accuracy'])  # Monitor accuracy during training
    return model

# Main execution block
if __name__ == '__main__':
    # Set up data paths
    base_path = os.path.join(os.getcwd(), 'human+activity+recognition+using+smartphones', 'UCI HAR Dataset', 'UCI HAR Dataset')
    
    # Load and prepare training data
    dataset = 'train'
    X_train, y_train, subjects_train = load_data(base_path)
    
    # Load and prepare test data
    dataset = 'test'
    X_test, y_test, subjects_test = load_data(base_path)
    
    # Preprocess the data for model input
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    # Convert labels to one-hot encoded format
    # Subtract 1 as labels are 1-indexed (1-6 instead of 0-5)
    y_train_cat = to_categorical(y_train - 1)
    y_test_cat = to_categorical(y_test - 1)
    
    # Initialize the model
    input_shape = (X_train_processed.shape[1], X_train_processed.shape[2])
    model = create_model(input_shape, n_classes=6)  # 6 activity classes
    
    # Configure training callbacks
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Number of epochs to wait before stopping
        restore_best_weights=True  # Restore model weights from the epoch with the best value
    )
    
    # Reduce learning rate when training plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Factor to reduce learning rate by
        patience=5,  # Number of epochs to wait before reducing LR
        min_lr=1e-6  # Minimum learning rate
    )
    
    # Train the model
    history = model.fit(X_train_processed, y_train_cat,
                       epochs=50,  # Maximum number of training epochs
                       batch_size=64,  # Number of samples per gradient update
                       validation_split=0.2,  # 20% of training data used for validation
                       callbacks=[early_stopping, reduce_lr],
                       verbose=1)
    
    # Evaluate model performance on test set
    test_loss, test_accuracy = model.evaluate(X_test_processed, y_test_cat, verbose=0)
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy metrics
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save training history plot
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Save the trained model
    model.save('har_cnn_lstm_model.h5')