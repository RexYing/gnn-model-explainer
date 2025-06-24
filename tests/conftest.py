import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

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


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object for testing."""
    class MockConfig:
        def __init__(self):
            self.dataset = "test_dataset"
            self.model = "test_model"
            self.batch_size = 32
            self.epochs = 10
            self.lr = 0.001
            self.hidden_dim = 64
            self.output_dim = 10
            self.dropout = 0.5
            self.weight_decay = 0.0001
            self.seed = 42
            self.device = "cpu"
            self.log_dir = None
            self.save_dir = None
            
    return MockConfig()


@pytest.fixture
def sample_graph():
    """Create a sample NetworkX graph for testing."""
    if not NETWORKX_AVAILABLE:
        pytest.skip("NetworkX not available")
    
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    
    for node in G.nodes():
        if TORCH_AVAILABLE:
            G.nodes[node]['feat'] = torch.randn(10)
        else:
            import random
            G.nodes[node]['feat'] = [random.random() for _ in range(10)]
        G.nodes[node]['label'] = node % 2
    
    return G


@pytest.fixture
def sample_tensor_data():
    """Provide sample tensor data for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("Torch not available")
    
    return {
        'features': torch.randn(100, 10),
        'labels': torch.randint(0, 2, (100,)),
        'edge_index': torch.randint(0, 100, (2, 200)),
        'batch': torch.zeros(100, dtype=torch.long)
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("Torch not available")
    
    class MockModel(torch.nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    return MockModel()


@pytest.fixture
def capture_stdout():
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    
    class OutputCapture:
        def __enter__(self):
            sys.stdout = captured_output
            return self
        
        def __exit__(self, *args):
            sys.stdout = old_stdout
        
        def get_output(self):
            return captured_output.getvalue()
    
    return OutputCapture()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    
    random.seed(42)
    
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass
    
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("Torch not available")
    
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randint(0, 2, (100,))
    )
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)


@pytest.fixture
def cleanup_files():
    """Cleanup helper to remove files created during tests."""
    files_to_cleanup = []
    
    def add_file(filepath):
        files_to_cleanup.append(filepath)
    
    yield add_file
    
    for filepath in files_to_cleanup:
        if os.path.exists(filepath):
            if os.path.isdir(filepath):
                shutil.rmtree(filepath)
            else:
                os.remove(filepath)


@pytest.fixture
def mock_logger(mocker):
    """Create a mock logger for testing logging functionality."""
    logger = mocker.MagicMock()
    logger.info = mocker.MagicMock()
    logger.debug = mocker.MagicMock()
    logger.warning = mocker.MagicMock()
    logger.error = mocker.MagicMock()
    return logger