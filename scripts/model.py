#!/usr/bin/env python3
"""
TOX21 NR-AR Classification Pipeline

A deep learning pipeline for predicting compound toxicity using the TOX21 Nuclear Receptor Androgen Receptor dataset.
Includes data preprocessing, model training, and evaluation components.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, 
                           confusion_matrix, 
                           roc_curve, 
                           auc, 
                           accuracy_score)
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers

# Constants
RANDOM_STATE = 42
DATA_PATH = "../data/Single/"
MODEL_PATH = "../models/"
OUTPUT_PATH = "../output/Single/NR-AR/"
RESULTS_FILE = "../output/Single/NR-AR/results.xlsx"

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print("Loading and preprocessing data...")
    
    # Load raw data
    df = pd.read_parquet(f"{DATA_PATH}/tox21_NR-AR_featurized.parquet")
    
    # Check initial class distribution
    print("\nInitial Class Distribution:")
    print(df['Y'].value_counts())
    print("\nClass Percentage:")
    print(df['Y'].value_counts(normalize=True) * 100)
    
    # Expand embeddings
    embeddings_df = pd.DataFrame(df["embedding"].tolist())
    embeddings_df.columns = [f"emb_{i}" for i in range(embeddings_df.shape[1])]
    df = df.drop(columns=["embedding"]).join(embeddings_df)
    
    return df

def split_data(df):
    """Split data into train/val/test sets."""
    print("\nSplitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['Y'], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Y'], random_state=RANDOM_STATE)
    
    # Save splits
    train_df.to_parquet(f"{DATA_PATH}/tox21_NR-AR_train.parquet")
    val_df.to_parquet(f"{DATA_PATH}/tox21_NR-AR_val.parquet")
    test_df.to_parquet(f"{DATA_PATH}/tox21_NR-AR_test.parquet")
    
    return train_df, val_df, test_df

def balance_data(train_df):
    """Apply SMOTE to balance class distribution."""
    print("\nBalancing training data with SMOTE...")
    X_train = train_df.filter(like="emb_")
    y_train = train_df["Y"]
    
    print("Original class distribution:", Counter(y_train))
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print("New class distribution:", Counter(y_resampled))
    
    df_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    df_resampled["Y"] = y_resampled
    df_resampled.to_parquet(f"{DATA_PATH}/tox21_NR-AR_train_resampled.parquet")
    
    return df_resampled

def visualize_data(train_df, resampled_df):
    """Generate data visualization plots."""
    print("\nGenerating visualizations...")
    
    # Class distribution plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x=train_df["Y"])
    plt.title("Class Distribution Before SMOTE")
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=resampled_df["Y"])
    plt.title("Class Distribution After SMOTE")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/class_distributions.png")
    plt.close()

def build_model(input_shape):
    """Build and compile the neural network model."""
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = optimizers.Adam(learning_rate=0.00001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, run_id):
    """Train and evaluate the model."""
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=1024,
        verbose=1
    )
    
    # Save model
    model.save(f"{MODEL_PATH}/{run_id}_tox21_classifier.h5")
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTrain Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    return history, (train_loss, train_acc), (val_loss, val_acc), (test_loss, test_acc)

def generate_plots(history, X_train, y_train, X_val, y_val, X_test, y_test, model, run_id):
    """Generate evaluation plots."""
    print("\nGenerating evaluation plots...")
    
    # Training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{run_id}/training_history.png")
    plt.close()
    
    # Confusion matrices
    datasets = {
        "Train": (X_train, y_train),
        "Validation": (X_val, y_val),
        "Test": (X_test, y_test)
    }
    
    plt.figure(figsize=(15, 4))
    for i, (name, (X, y)) in enumerate(datasets.items(), 1):
        y_pred = (model.predict(X) > 0.5).astype("int32")
        cm = confusion_matrix(y, y_pred)
        
        plt.subplot(1, 3, i)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["Negative", "Positive"],
                   yticklabels=["Negative", "Positive"])
        plt.title(f"Confusion Matrix ({name})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{run_id}/confusion_matrices.png")
    plt.close()
    
    # ROC curves
    plt.figure(figsize=(8, 6))
    for X, y, label in [(X_train, y_train, "Train"),
                        (X_val, y_val, "Validation"),
                        (X_test, y_test, "Test")]:
        y_prob = model.predict(X)
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(f"{OUTPUT_PATH}/{run_id}/roc_curves.png")
    plt.close()

def main():
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    # Data pipeline
    df = load_and_preprocess_data()
    train_df, val_df, test_df = split_data(df)
    resampled_df = balance_data(train_df)
    visualize_data(train_df, resampled_df)
    
    # Prepare features
    X_train = resampled_df.drop(columns=["Y"]).select_dtypes(include=[float, int])
    y_train = resampled_df["Y"]
    
    X_val = val_df.drop(columns=["Y"]).select_dtypes(include=[float, int])
    y_val = val_df["Y"]
    
    X_test = test_df.drop(columns=["Y"]).select_dtypes(include=[float, int])
    y_test = test_df["Y"]
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Get next run ID
    dfr = pd.read_excel(RESULTS_FILE, engine='openpyxl')
    run_id = len(dfr) + 1
    
    # Create output directory
    os.makedirs(f"{OUTPUT_PATH}/{run_id}", exist_ok=True)
    
    # Model pipeline
    model = build_model(X_train.shape[1])
    history, train_metrics, val_metrics, test_metrics = train_and_evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test, run_id
    )
    generate_plots(history, X_train, y_train, X_val, y_val, X_test, y_test, model, run_id)
    
    # Generate classification reports
    def get_report(X, y):
        y_pred = (model.predict(X) > 0.5).astype("int32")
        return classification_report(y, y_pred, output_dict=True)
    
    train_report = get_report(X_train, y_train)
    val_report = get_report(X_val, y_val)
    test_report = get_report(X_test, y_test)
    
    # Log results
    results = {
        "Number": run_id,
        "Model": "DNN",
        "Dataset": "TOX21-NR_AR",
        "Hidden Layers": 4,
        "Hidden Layer Sizes": "[512, 256, 128, 64]",
        "Activation Function": "ReLU + Sigmoid",
        "Train Loss": train_metrics[0],
        "Validation Loss": val_metrics[0],
        "Test Loss": test_metrics[0],
        "Epochs": 150,
        "Batch Size": 1024,
        "Learning Rate": 0.00001,
        "Train Accuracy": train_metrics[1],
        "Validation Accuracy": val_metrics[1],
        "Test Accuracy": test_metrics[1],
        "Optimizer": "Adam",
        "Train Classification Report": str(train_report),
        "Validation Classification Report": str(val_report),
        "Test Classification Report": str(test_report)
    }
    
    results_df = pd.DataFrame([results])
    if os.path.exists(RESULTS_FILE):
        existing_df = pd.read_excel(RESULTS_FILE)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    
    results_df.to_excel(RESULTS_FILE, index=False)
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()