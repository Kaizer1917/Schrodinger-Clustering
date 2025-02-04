### Core Components

1. **SAMBA Model**
   - Graph-Mamba architecture for stock price prediction
   - Combines graph neural networks with Mamba sequence modeling
   - Gaussian kernel graph construction for dynamic relationship modeling

2. **Quantum-Inspired Clustering**
   - New SchrodingerClustering implementation based on quantum mechanics principles
   - Uses quantum state evolution for dimensionality reduction
   - Combines Hamiltonian dynamics with K-means clustering
   - Enables discovery of quantum-inspired market states

### Dataset Structure

The Dataset directory contains three CSV files for different market indices:
- `combined_dataframe_DJI.csv`: Dow Jones Industrial Average data
- `combined_dataframe_IXIC.csv`: NASDAQ data
- `combined_dataframe_NYSE.csv`: New York Stock Exchange data

Each dataset contains 82 daily stock features from January 2010 to November 2023.

### Usage

```python
from samba_combined import SAMBAWithClustering, ModelArgs

# Initialize model with clustering
model = SAMBAWithClustering(
    ModelArgs=model_args,
    hidden=128,
    inp=82,  # number of features
    out=1,   # prediction target
    embed=64,
    cheb_k=3,
    n_clusters=8  # number of market state clusters
)

# Train model and get predictions with cluster assignments
predictions, clusters = cluster_analysis(model, data_loader)
```

### Key Features

1. **Unified Architecture**
   - Single file implementation
   - Seamless integration of all components
   - Easy to understand and modify

2. **Enhanced Functionality**
   - Added quantum-inspired clustering
   - Market state analysis capabilities
   - Improved data preprocessing

3. **Performance Metrics**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Square Error)
   - Clustering analysis metrics
