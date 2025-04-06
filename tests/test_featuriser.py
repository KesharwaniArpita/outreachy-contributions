import pytest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys


# Go up one level to the project root, then into scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Test fixtures
@pytest.fixture
def mock_data(tmp_path):
    """Create a dummy parquet file with SMILES and labels"""
    data = {
        "smiles": ["CCO", "CCN", "C=O"], 
        "label": [0, 1, 0]
    }
    df = pd.DataFrame(data)
    path = tmp_path / "test_data.parquet"
    df.to_parquet(path)
    return path

@pytest.fixture
def mock_ersilia():
    """Mock Ersilia model with fixed embeddings"""
    mock = MagicMock()
    mock.transform.return_value = [
        [0.1, 0.2, 0.3],  # Mock embedding for "CCO"
        [0.4, 0.5, 0.6],  # Mock embedding for "CCN"
        [0.7, 0.8, 0.9]   # Mock embedding for "C=O"
    ]
    return mock

def test_featurisation_output_structure(mock_data, mock_ersilia, tmp_path):
    """Verify output file has correct structure"""
    from featuriser import featurise_and_save
    
    output_path = tmp_path / "output.parquet"
    
    with patch("eosce.models.ErsiliaCompoundEmbeddings", return_value=mock_ersilia):
        featurise_and_save(mock_data, output_path, batch_size=2)
    
    # Verify output file exists
    assert output_path.exists()
    
    # Verify content structure
    df = pd.read_parquet(output_path)
    assert set(df.columns) == {"embedding", "label"}
    assert len(df) == 3  # 3 input samples
    assert isinstance(df["embedding"].iloc[0], list)  # Embeddings should be lists

def test_batch_processing(mock_data, mock_ersilia):
    """Verify batches are processed correctly"""
    from featuriser import featurise_and_save
    
    with patch("eosce.models.ErsiliaCompoundEmbeddings", return_value=mock_ersilia):
        # Test with batch_size=2 (should process in 2 batches)
        mock_ersilia.transform.reset_mock()
        featurise_and_save(mock_data, "dummy.parquet", batch_size=2)
        
        # Should be called twice (batch_size=2 for 3 samples)
        assert mock_ersilia.transform.call_count == 2

def test_ersilia_integration(mock_data, mock_ersilia):
    """Verify Ersilia model receives correct inputs"""
    from featuriser import featurise_and_save
    
    with patch("eosce.models.ErsiliaCompoundEmbeddings", return_value=mock_ersilia):
        featurise_and_save(mock_data, "dummy.parquet")
        
        # Verify first call received correct SMILES
        args, _ = mock_ersilia.transform.call_args
        assert set(args[0]) == {"CCO", "CCN", "C=O"}

def test_empty_input(tmp_path):
    """Test handling of empty input file"""
    from featuriser import featurise_and_save
    
    empty_path = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(empty_path)
    
    with pytest.raises(ValueError):
        featurise_and_save(empty_path, "dummy.parquet")

def test_output_embeddings(mock_data, mock_ersilia, tmp_path):
    """Verify embeddings are correctly stored"""
    from featuriser import featurise_and_save
    
    output_path = tmp_path / "output.parquet"
    
    with patch("eosce.models.ErsiliaCompoundEmbeddings", return_value=mock_ersilia):
        featurise_and_save(mock_data, output_path)
    
    df = pd.read_parquet(output_path)
    # Verify mock embeddings were stored correctly
    assert df["embedding"].tolist() == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6], 
        [0.7, 0.8, 0.9]
    ]