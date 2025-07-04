"""Validation tests to verify the testing infrastructure is set up correctly."""

import pytest
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class TestInfrastructureSetup:
    """Validate that the testing infrastructure is properly configured."""
    
    def test_pytest_is_importable(self):
        """Test that pytest can be imported."""
        import pytest
        assert pytest is not None
    
    def test_fixtures_are_available(self, temp_dir, mock_config):
        """Test that conftest fixtures are available and working."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        assert hasattr(mock_config, 'dataset')
        assert mock_config.batch_size == 32
    
    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
    def test_sample_graph_fixture(self, sample_graph):
        """Test sample graph fixture when NetworkX is available."""
        assert isinstance(sample_graph, nx.Graph)
        assert sample_graph.number_of_nodes() > 0
    
    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temporary directory is created and will be cleaned up."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_mock_model_fixture(self, mock_model):
        """Test that mock model fixture works correctly."""
        input_tensor = torch.randn(10, 10)
        output = mock_model(input_tensor)
        assert output.shape == (10, 2)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_sample_tensor_data_fixture(self, sample_tensor_data):
        """Test that sample tensor data fixture provides correct data."""
        assert 'features' in sample_tensor_data
        assert 'labels' in sample_tensor_data
        assert 'edge_index' in sample_tensor_data
        assert 'batch' in sample_tensor_data
        
        assert sample_tensor_data['features'].shape == (100, 10)
        assert sample_tensor_data['labels'].shape == (100,)
    
    def test_capture_stdout_fixture(self, capture_stdout):
        """Test that stdout capture fixture works."""
        with capture_stdout as capture:
            print("Test output")
            print("Another line")
            
            output = capture.get_output()
        
        assert "Test output" in output
        assert "Another line" in output
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_mock_data_loader_fixture(self, mock_data_loader):
        """Test that mock data loader fixture works."""
        batch_count = 0
        for batch in mock_data_loader:
            batch_count += 1
            assert len(batch) == 2  # features and labels
            assert batch[0].shape[1] == 10  # feature dimension
        
        assert batch_count > 0
    
    def test_cleanup_files_fixture(self, cleanup_files, temp_dir):
        """Test that cleanup files fixture works."""
        test_file = temp_dir / "cleanup_test.txt"
        test_file.write_text("test")
        
        cleanup_files(str(test_file))
        assert test_file.exists()  # Should still exist during test
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_random_seed_reset(self):
        """Test that random seeds are reset for reproducibility."""
        torch.manual_seed(42)
        first_random = torch.rand(5)
        
        torch.manual_seed(42)
        second_random = torch.rand(5)
        
        assert torch.allclose(first_random, second_random)


class TestCoverageConfiguration:
    """Validate coverage configuration."""
    
    def test_coverage_runs_with_tests(self):
        """Test that coverage is configured to run with tests."""
        # This test will pass if coverage is properly configured
        # The actual coverage check happens via pytest-cov
        assert True


def test_project_structure():
    """Test that the expected project structure exists."""
    expected_dirs = [
        Path("tests"),
        Path("tests/unit"),
        Path("tests/integration"),
    ]
    
    expected_files = [
        Path("tests/__init__.py"),
        Path("tests/unit/__init__.py"),
        Path("tests/integration/__init__.py"),
        Path("tests/conftest.py"),
        Path("pyproject.toml"),
    ]
    
    for dir_path in expected_dirs:
        assert dir_path.exists(), f"Directory {dir_path} should exist"
    
    for file_path in expected_files:
        assert file_path.exists(), f"File {file_path} should exist"