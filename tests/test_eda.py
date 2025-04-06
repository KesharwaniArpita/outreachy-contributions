import os
import pytest
import pandas as pd
from scripts.eda import perform_eda
import matplotlib.pyplot as plt
from pathlib import Path
import sys


# Go up one level to the project root, then into scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

@pytest.fixture
def sample_data(tmp_path):
    """Fixture creating synthetic test data"""
    data = {
        'feature1': [1, 2, 3, None, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    output_dir = tmp_path / "eda_output"
    return df, output_dir

def test_eda_output_files(sample_data):
    """Verify EDA generates all expected files"""
    df, output_dir = sample_data
    perform_eda(df, output_dir)
    
    expected_files = [
        "eda_report.txt",
        "summary_statistics.csv",
        "missing_values_heatmap.png",
        "feature_distributions.png",
        "boxplot_outliers.png"
    ]
    
    for file in expected_files:
        assert (output_dir / file).exists(), f"Missing {file}"

def test_eda_report_content(sample_data):
    """Check report contains critical sections"""
    df, output_dir = sample_data
    perform_eda(df, output_dir)
    
    with open(output_dir / "eda_report.txt") as f:
        content = f.read()
        
    assert "Basic Information:" in content
    assert "First 5 Rows:" in content
    assert "Missing Values:" in content
    assert "feature1    1" in content  # Verify NaN count appears

def test_missing_values_heatmap(sample_data):
    """Validate heatmap generation"""
    df, output_dir = sample_data
    perform_eda(df, output_dir)
    
    img = plt.imread(output_dir / "missing_values_heatmap.png")
    assert img.shape == (480, 640, 3)  # Default figsize dimensions

def test_summary_stats(sample_data):
    """Check statistics file structure"""
    df, output_dir = sample_data
    perform_eda(df, output_dir)
    
    stats = pd.read_csv(output_dir / "summary_statistics.csv", index_col=0)
    assert set(stats.index) == {'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'}
    assert "feature2" in stats.columns

def test_empty_data(tmp_path):
    """Test edge case: empty dataframe"""
    df = pd.DataFrame()
    output_dir = tmp_path / "empty_test"
    
    with pytest.raises(ValueError):
        perform_eda(df, output_dir)

def test_visualizations(tmp_path):
    """Verify visualization files are valid images"""
    df = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6]})
    output_dir = tmp_path / "viz_test"
    perform_eda(df, output_dir)
    
    for viz_file in ["feature_distributions.png", "boxplot_outliers.png"]:
        try:
            plt.imread(output_dir / viz_file)
        except Exception as e:
            pytest.fail(f"Corrupted image file: {viz_file} ({str(e)})")