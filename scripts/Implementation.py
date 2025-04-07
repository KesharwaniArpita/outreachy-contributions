import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate a trained model on a new dataset.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model (.h5 file)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the evaluation data (.parquet file)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/', help='Output directory for results')
    parser.add_argument('--target_col', type=str, default='Y', help='Name of the target column')
    parser.add_argument('--embedding_col', type=str, default='embedding', help='Name of the embedding column')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for evaluation')
    return parser.parse_args()

def load_and_preprocess_data(filepath, target_col, embedding_col):
    """Load and preprocess the dataset"""
    df = pd.read_parquet(filepath)
    
    # Print class distribution
    print("\nClass Distribution:")
    print(df[target_col].value_counts())
    print("\nClass Percentage:")
    print(df[target_col].value_counts(normalize=True) * 100)
    
    # Expand embeddings if present
    if embedding_col in df.columns:
        embeddings_df = pd.DataFrame(df[embedding_col].tolist())
        embeddings_df.columns = [f"emb_{i}" for i in range(embeddings_df.shape[1])]
        df = df.drop(columns=[embedding_col]).join(embeddings_df)
    
    # Separate features and target
    X = df.drop(columns=[target_col]).select_dtypes(include=[float, int])
    y = df[target_col]
    
    return X, y

def evaluate_model(model, X, y, dataset_name, output_dir, batch_size):
    """Evaluate model and generate visualizations"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Predictions
    y_pred_prob = model.predict(X, batch_size=batch_size)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    
    # Calculate metrics
    results = {
        'dataset': dataset_name,
        'loss': float(model.evaluate(X, y, verbose=0, batch_size=batch_size)[0]),
        'accuracy': float(model.evaluate(X, y, verbose=0, batch_size=batch_size)[1]),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        'classification_report': classification_report(y, y_pred, output_dict=True),
        'roc_auc': float(auc(*roc_curve(y, y_pred_prob)[:2]))
    }
    
    # Save metrics to JSON
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate visualizations
    plot_confusion_matrix(results['confusion_matrix'], dataset_name, output_dir)
    plot_roc_curve(y, y_pred_prob, dataset_name, output_dir)
    plot_tsne(X, y, y_pred, dataset_name, output_dir)
    
    return results

def plot_confusion_matrix(cm, dataset_name, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(f"Confusion Matrix ({dataset_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{output_dir}/{dataset_name}_confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, dataset_name, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({dataset_name})")
    plt.legend()
    plt.savefig(f"{output_dir}/{dataset_name}_roc_curve.png")
    plt.close()

def plot_tsne(X, y_true, y_pred, dataset_name, output_dir):
    """Plot and save t-SNE visualizations"""
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # True labels
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_true, 
                        palette="coolwarm", alpha=0.7)
        plt.title(f"t-SNE - {dataset_name} (True Labels)")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.legend(title="Label", loc="best")
        plt.savefig(f"{output_dir}/{dataset_name}_tsne_true.png")
        plt.close()
        
        # Predicted labels
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_pred.flatten(),
                        palette="coolwarm", alpha=0.7)
        plt.title(f"t-SNE - {dataset_name} (Predicted Labels)")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.legend(title="Label", loc="best")
        plt.savefig(f"{output_dir}/{dataset_name}_tsne_pred.png")
        plt.close()
    except Exception as e:
        print(f"Could not generate t-SNE plots: {str(e)}")

def main():
    args = parse_arguments()
    
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    print(f"\nLoading and preprocessing data from {args.data_path}...")
    X, y = load_and_preprocess_data(args.data_path, args.target_col, args.embedding_col)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    print(f"\nEvaluating model on {dataset_name}...")
    results = evaluate_model(model, X_scaled, y, dataset_name, args.output_dir, args.batch_size)
    
    print("\nEvaluation Results:")
    print(f"- Loss: {results['loss']:.4f}")
    print(f"- Accuracy: {results['accuracy']:.4f}")
    print(f"- ROC AUC: {results['roc_auc']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()