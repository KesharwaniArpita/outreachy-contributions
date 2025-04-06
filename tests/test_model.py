import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import tensorflow as tf
import sys

# Go up one level to the project root, then into scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Constants for testing
TEST_DATA_SHAPE = (100, 1024)  # 100 samples, 1024 features
TEST_DATA_PATH = "tests/test_data/"

@pytest.fixture
def mock_data():
    """Create mock dataframe with embeddings and labels"""
    np.random.seed(42)
    data = {
        "embedding": [list(np.random.rand(1024)) for _ in range(100)],
        "Y": [0] * 90 + [1] * 10  # 10% positive class
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_train_data():
    """Mock training data after SMOTE"""
    return pd.DataFrame({
        "emb_0": np.random.rand(200),
        "emb_1": np.random.rand(200),
        "Y": [0] * 100 + [1] * 100
    })

def test_load_and_preprocess_data(mock_data, tmp_path):
    """Test data loading and preprocessing"""
    from model import load_and_preprocess_data
    
    # Create test parquet file
    test_file = tmp_path / "test_data.parquet"
    mock_data.to_parquet(test_file)
    
    # Patch the DATA_PATH
    with patch("model.DATA_PATH", str(tmp_path) + "/"):
        df = load_and_preprocess_data()
    
    # Verify output
    assert isinstance(df, pd.DataFrame)
    assert "emb_0" in df.columns  # First embedding column
    assert "Y" in df.columns  # Target column
    assert df.shape[0] == 100  # Same number of samples

def test_split_data(mock_data):
    """Test data splitting functionality"""
    from model import split_data
    
    # Add some dummy embedding columns
    for i in range(5):
        mock_data[f"emb_{i}"] = np.random.rand(len(mock_data))
    
    train_df, val_df, test_df = split_data(mock_data)
    
    # Verify splits
    assert len(train_df) > len(val_df)  # 80% train
    assert len(val_df) == len(test_df)  # 10% val, 10% test
    assert set(train_df.columns) == set(val_df.columns) == set(test_df.columns)

def test_balance_data():
    """Test SMOTE balancing"""
    from model import balance_data
    
    # Create imbalanced data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"emb_{i}" for i in range(5)])
    y = pd.Series([0] * 90 + [1] * 10)  # 10% positive
    imbalanced_df = X.assign(Y=y)
    
    balanced_df = balance_data(imbalanced_df)
    
    # Verify balancing
    counts = balanced_df["Y"].value_counts()
    assert counts[0] == counts[1]  # Should be balanced now

def test_build_model():
    """Test model construction"""
    from model import build_model
    
    model = build_model(input_shape=1024)
    
    # Verify model structure
    assert len(model.layers) == 11  # Input + (Dense+BN+Dropout)*4 + Output
    assert model.layers[-1].activation.__name__ == "sigmoid"  # Final activation
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

def test_train_and_evaluate(mock_train_data):
    """Test training workflow"""
    from model import train_and_evaluate
    
    # Create mock model
    model = MagicMock()
    model.evaluate.side_effect = [
        (0.5, 0.8),  # train metrics
        (0.6, 0.75),  # val metrics
        (0.7, 0.7)    # test metrics
    ]
    
    # Mock data
    X_train = np.random.rand(100, 1024)
    y_train = np.random.randint(0, 2, 100)
    
    # Test training
    with patch("model.build_model", return_value=model):
        history, train_metrics, val_metrics, test_metrics = train_and_evaluate(
            model, X_train, y_train, X_train, y_train, X_train, y_train, 1
        )
    
    # Verify metrics
    assert train_metrics == (0.5, 0.8)
    assert model.fit.call_count == 1

def test_generate_plots():
    """Test plot generation (smoke test)"""
    from model import generate_plots
    
    # Mock model and data
    model = MagicMock()
    model.predict.side_effect = [
        np.random.rand(100, 1),  # train
        np.random.rand(100, 1),  # val
        np.random.rand(100, 1)   # test
    ]
    
    # Mock history
    history = {
        "accuracy": [0.7, 0.8, 0.9],
        "val_accuracy": [0.6, 0.7, 0.8],
        "loss": [0.5, 0.4, 0.3],
        "val_loss": [0.6, 0.5, 0.4]
    }
    
    # Test plot generation
    with patch("matplotlib.pyplot.savefig"):
        generate_plots(
            history,
            np.random.rand(100, 1024), np.random.randint(0, 2, 100),
            np.random.rand(100, 1024), np.random.randint(0, 2, 100),
            np.random.rand(100, 1024), np.random.randint(0, 2, 100),
            model, 1
        )
    
    # Verify plot calls
    assert model.predict.call_count == 3

def test_full_pipeline(mock_data, tmp_path):
    """Smoke test for full pipeline"""
    from model import main
    
    # Setup test environment
    os.makedirs(f"{tmp_path}/data/Single", exist_ok=True)
    os.makedirs(f"{tmp_path}/models", exist_ok=True)
    os.makedirs(f"{tmp_path}/output/Single/NR-AR", exist_ok=True)
    
    # Create test data file
    mock_data.to_parquet(f"{tmp_path}/data/Single/tox21_NR-AR_featurized.parquet")
    
    # Create empty results file
    pd.DataFrame().to_excel(f"{tmp_path}/output/Single/NR-AR/results.xlsx")
    
    # Patch paths
    with patch("model.DATA_PATH", f"{tmp_path}/data/Single/"):
        with patch("model.MODEL_PATH", f"{tmp_path}/models/"):
            with patch("model.OUTPUT_PATH", f"{tmp_path}/output/Single/NR-AR/"):
                with patch("model.RESULTS_FILE", f"{tmp_path}/output/Single/NR-AR/results.xlsx"):
                    # Run with reduced epochs for testing
                    with patch("model.build_model") as mock_build:
                        mock_model = MagicMock()
                        mock_model.evaluate.side_effect = [
                            (0.5, 0.9),  # train
                            (0.6, 0.8),  # val
                            (0.7, 0.7)   # test
                        ]
                        mock_build.return_value = mock_model
                        
                        # Run pipeline with 2 epochs for testing
                        with patch("model.train_and_evaluate") as mock_train:
                            mock_train.return_value = (
                                {"accuracy": [0.1], "loss": [0.1]},  # history
                                (0.1, 0.9),  # train
                                (0.2, 0.8),  # val
                                (0.3, 0.7)   # test
                            )
                            main()
    
    # Verify outputs were created
    assert os.path.exists(f"{tmp_path}/models/1_tox21_classifier.h5")
    assert os.path.exists(f"{tmp_path}/output/Single/NR-AR/1/training_history.png")