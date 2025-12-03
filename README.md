# Octree-Based 3D Occupancy Mapping
### Carnegie Mellon University — 16-362 Mobile Robot Programming Lab

This project implements a full 3D occupancy Octree mapping pipeline using **pure Python** and compares it to a **pure Python dense voxel grid** implementation. Both implementations are written from scratch in Python for fair comparison.

The project includes:

- **Pure Python Octree** with probabilistic occupancy (log-odds update)
- **Pure Python Dense Voxel Grid** (sparse dictionary-based implementation)
- Point cloud loading and preprocessing (Open3D for I/O only)
- Synthetic sparse world generation for testing scalability
- Matplotlib-based visualizations (no Open3D GUI needed)
- Comprehensive benchmarking across multiple Octree depths
- Detailed runtime, memory, and efficiency comparison plots
- Theoretical dense grid analysis for sparse world scenarios

This README explains the directory structure, usage, technical specifications, and purpose of each file.

---

## Project Structure

```text
16362-OctreeMapping/
│
├── data/                        # Input point cloud data (.pcd)
│     └── (auto-downloaded by Open3D on first run)
│
├── results/                     # Output plots and benchmark results
│     ├── runtime_vs_depth*.png          # Runtime comparisons
│     ├── memory_vs_depth*.png           # Memory usage comparisons
│     ├── octree_resolution*.png         # Node growth analysis
│     ├── efficiency_comparison*.png     # Comprehensive efficiency plots
│     ├── dataset_comparison.png         # Real vs synthetic comparison
│     ├── theoretical_dense_comparison.png  # Full grid vs octree analysis
│     └── runtime_comparison*.png        # Direct runtime bar charts
│
├── src/
│     ├── main.py                # Main script: runs experiments on real + synthetic data
│     ├── benchmark.py           # Comprehensive benchmarking across depths
│     ├── runtime_comparison.py  # Standalone runtime comparison generator
│     ├── mapping.py             # Point cloud loading + Octree building
│     ├── octree.py              # Pure Python Octree implementation
│     ├── compare.py             # Pure Python Dense Voxel Grid implementation
│     ├── visualize.py           # Matplotlib 3D visualization for Octree
│     ├── metrics.py             # Metrics: node counts, bbox, memory estimation
│     └── plots.py               # Plot generation utilities
│
├── venv/                        # Python virtual environment
│
└── README.md                    # This documentation file
```

-----

## System Requirements

### Operating System
- **macOS** (tested on macOS 14+)
- **Linux** (Ubuntu 20.04+ recommended)
- **Windows** (with WSL2 or native Python support)

### Python Version
- **Python 3.9** or higher (3.10+ recommended)
- Python 3.10 is the primary development version
- Python 3.11 and 3.12 are fully supported

### Hardware Requirements
- **RAM**: Minimum 4GB, recommended 8GB+ for large point clouds
- **Storage**: ~500MB for dependencies and results
- **CPU**: Any modern processor (benchmarking benefits from multi-core)

### Software Dependencies
- Python 3.9+
- pip (Python package manager)
- Virtual environment support (venv module)

## Installation & Setup

### 1. Create and activate a Python virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

**Alternative Python versions:**
```bash
python3.9 -m venv venv    # Python 3.9
python3.11 -m venv venv   # Python 3.11
python3 -m venv venv      # System default
```

### 2. Install required packages

```bash
pip install open3d matplotlib numpy
```

**Package Versions:**
- `open3d`: 0.17.0+ (used only for point cloud I/O)
- `matplotlib`: 3.5.0+ (for all visualizations)
- `numpy`: 1.21.0+ (for numerical operations)

**Note:** Open3D GUI is disabled via environment variable. Only Matplotlib visualizations are used.

### 3. Verify Installation

```bash
python -c "import open3d; import numpy; import matplotlib; print('All dependencies installed successfully!')"
```

### Environment Notes

> **Using a different Python version?**  
> The project was built with Python 3.10, but Python 3.9–3.12 all work. If `python3.10` is unavailable, create the virtual env with whichever version you have.

> **Can't install `open3d`?**  
> Install the latest compatible version: `pip install open3d`. The code only relies on point-cloud I/O functionality, which is stable across versions.

> **macOS/Linux activation:**
> ```bash
> source venv/bin/activate
> ```

> **Windows activation:**
> ```bash
> venv\Scripts\activate
> ```

-----

## Running the Project

### Run Main Experiments

```bash
python src/main.py
```

This script runs two experiments:

1. **Real Dataset (fragment.pcd)**:
   - Loads point cloud from Open3D's sample dataset
   - Builds pure Python dense voxel grid (sparse dictionary)
   - Builds pure Python Octree (depth = 6)
   - Prints comprehensive metrics and comparative analysis
   - Generates runtime comparison graphs

2. **Synthetic Sparse World**:
   - Generates a large 100m³ world with sparse occupancy
   - 200 small object clusters (300 points each)
   - Demonstrates octree advantages on sparse data
   - Shows theoretical dense grid vs actual sparse structures

**Output:**
- Console metrics and analysis
- Runtime comparison plots saved to `results/`
- Optional: Matplotlib 3D visualizations (can be disabled)

### Generate Runtime Comparison Graphs

```bash
python src/runtime_comparison.py
```

Generates focused runtime comparison graphs:
- Bar chart comparison at depth 6
- Runtime vs depth line plot

-----

## Running Comprehensive Benchmarks

To benchmark performance across multiple Octree depths on both real and synthetic datasets:

```bash
python src/benchmark.py
```

This script automatically:

1. **Tests both datasets**:
   - Real dataset (fragment.pcd)
   - Synthetic sparse world (100m³, sparse occupancy)

2. **Tests multiple depths**: `[3, 4, 5, 6, 7]`

3. **Measures**:
   - Runtime (seconds)
   - Node count (total + occupied leaves)
   - Estimated memory usage
   - Build time per node/cell
   - Memory per node/cell
   - Theoretical dense grid comparisons

4. **Generates plots** (saved to `results/`):
   - Individual dataset plots:
     - `runtime_vs_depth_real.png` / `runtime_vs_depth_sparse.png`
     - `memory_vs_depth_real.png` / `memory_vs_depth_sparse.png`
     - `octree_resolution_real.png` / `octree_resolution_sparse.png`
     - `efficiency_comparison_real.png` / `efficiency_comparison_sparse.png`
   - Comparison plots:
     - `dataset_comparison.png` (6-panel real vs sparse comparison)
     - `theoretical_dense_comparison.png` (theoretical full grid analysis)

5. **Prints summary statistics**:
   - Best octree depth by memory
   - Compression ratios
   - Memory efficiency gains

-----

## Technical Specifications

### Implementation Details

#### **Octree (`octree.py`)**
- **Language**: Pure Python (no C++ dependencies)
- **Data Structure**: Recursive tree with 8 children per node
- **Occupancy Model**: Probabilistic log-odds
  - Occupied: `log(0.7 / 0.3) ≈ 0.847`
  - Free: `log(0.4 / 0.6) ≈ -0.405`
  - Clamped to `[-3.5, 3.5]` range
- **Insertion Algorithm**: Depth-limited tree traversal
  - Traverses from root to target depth
  - Subdivides nodes on-demand (lazy subdivision)
  - Updates log-odds at leaf nodes
- **Memory Estimate**: ~200 bytes per node
  - Center (3 floats): 24 bytes
  - Size (1 float): 8 bytes
  - Log-odds (1 float): 8 bytes
  - Children pointers (8 pointers): 64 bytes
  - Python object overhead: ~96 bytes

#### **Dense Voxel Grid (`compare.py`)**
- **Language**: Pure Python (sparse dictionary implementation)
- **Data Structure**: Dictionary `{(i, j, k): count}`
- **Voxel Indexing**: Integer grid coordinates
  - Converts 3D point → voxel indices via floor division
  - Stores only occupied voxels (sparse storage)
- **Memory Estimate**: ~50 bytes per occupied voxel
  - Voxel index tuple (3 ints): 24 bytes
  - Point count (1 int): 8 bytes
  - Dictionary overhead: ~18 bytes
- **Theoretical Full Grid**: `(bbox_size / voxel_size)³` cells
  - Only computed for comparison, not allocated

### Algorithm Complexity

| Operation | Dense Voxel Grid | Octree |
|-----------|------------------|--------|
| **Point Insertion** | O(1) per point | O(depth) per point |
| **Memory (sparse)** | O(occupied_voxels) | O(total_nodes) |
| **Memory (full grid)** | O(bbox³) | O(occupied_regions) |
| **Query** | O(1) hash lookup | O(depth) tree traversal |

### Key Files Explained

#### `octree.py`
- **Core Octree Implementation**
- `OctreeNode`: Tree node with center, size, children, log-odds
- `insert_point()`: Depth-limited insertion with subdivision
- `update_log_odds()`: Bayesian occupancy update
- `subdivide()`: Creates 8 child nodes

#### `compare.py`
- **Pure Python Dense Voxel Grid**
- `DenseVoxelGrid`: Sparse dictionary-based voxel grid
- `_point_to_voxel_index()`: Converts 3D point to grid coordinates
- `_build_grid()`: Inserts all points into sparse dictionary
- `estimate_memory_bytes()`: Memory estimation for occupied voxels

#### `mapping.py`
- **Point Cloud Processing**
- `load_point_cloud()`: Loads `.pcd` files via Open3D
- `build_octree()`: Constructs octree from point array

#### `metrics.py`
- **Analysis Utilities**
- `count_octree_nodes()`: Total node count (DFS traversal)
- `count_octree_occupied_leaves()`: Occupied leaf count
- `estimate_octree_memory()`: Memory estimation
- `bbox()`: Bounding box computation
- `print_metrics()`: Formatted console output

#### `visualize.py`
- **3D Visualization**
- `draw_octree()`: Matplotlib 3D scatter plot of occupied leaves
- `collect_voxels()`: Recursive voxel collection

#### `plots.py`
- **Plot Generation**
- `plot_runtime_comparison()`: Bar chart + speedup visualization
- `plot_runtime()`: Runtime vs depth line plot
- `plot_memory()`: Memory usage comparison
- `plot_efficiency_comparison()`: 4-panel comprehensive analysis
- `plot_dataset_comparison()`: 6-panel real vs sparse comparison
- `plot_theoretical_dense_comparison()`: Theoretical full grid analysis

#### `main.py`
- **Main Experiment Script**
- `generate_sparse_world()`: Creates synthetic sparse test data
- `run_experiment()`: Runs dense grid + octree comparison
- Tests both real and synthetic datasets

#### `benchmark.py`
- **Comprehensive Benchmarking**
- `run_single_benchmark()`: Benchmarks single dataset across depths
- `run_benchmarks()`: Runs full benchmark suite on both datasets
- Generates all comparison plots

#### `runtime_comparison.py`
- **Focused Runtime Analysis**
- `generate_runtime_comparison_graphs()`: Generates runtime-focused plots

-----

## Performance Characteristics

### Runtime Performance

**For Smaller/Denser Datasets:**
- **Dense Voxel Grid is faster** (typically 2-5x)
  - O(1) dictionary insertion per point
  - No tree traversal overhead
  - Direct hash-based lookup

**For Larger/Sparser Datasets:**
- **Octree becomes more efficient**
  - Adapts to sparse occupancy
  - Fewer nodes in empty regions
  - Better memory efficiency

### Memory Efficiency

- **Dense Voxel Grid (sparse)**: Stores only occupied voxels
  - Memory: `occupied_voxels × 50 bytes`
  - Theoretical full grid: `(bbox_size/voxel_size)³ × 50 bytes`

- **Octree**: Stores only explored regions
  - Memory: `total_nodes × 200 bytes`
  - Typically 2-10x fewer nodes than occupied voxels
  - Compression improves with sparsity

### Key Insights

1. **Both implementations are pure Python** for fair comparison
2. **Dense grid uses sparse dictionary** (not full 3D array)
3. **Octree overhead**: Tree traversal + subdivision checks
4. **Octree advantage**: Adaptive to sparse occupancy
5. **Memory estimates** are based on Python object sizes, not actual memory profiling

## Notes for Developers / Teammates

- **Both implementations are pure Python** (fair comparison, educational focus)
- **Dense voxel grid** uses sparse dictionary (not full 3D array allocation)
- **Visualization** is Matplotlib-only (avoids Open3D GUI issues on macOS)
- **All benchmarking plots** are reproducible via `benchmark.py`
- **Memory estimates** are approximations based on Python object overhead
- **Theoretical dense grid** calculations show potential memory savings
- **Synthetic sparse world** demonstrates octree advantages on large, sparse datasets

-----

## Output Files

### Generated Plots

All plots are saved to the `results/` directory:

**Runtime Analysis:**
- `runtime_comparison_*.png`: Bar chart comparisons with speedup ratios
- `runtime_vs_depth*.png`: Runtime vs octree depth line plots

**Memory Analysis:**
- `memory_vs_depth*.png`: Occupied cells vs depth (log scale)
- `efficiency_comparison*.png`: 4-panel comprehensive efficiency analysis

**Structure Analysis:**
- `octree_resolution*.png`: Total node growth vs depth
- `dataset_comparison.png`: 6-panel real vs synthetic comparison
- `theoretical_dense_comparison.png`: Theoretical full grid vs octree analysis

### Console Output

The scripts provide detailed console output including:
- Point cloud statistics
- Runtime measurements (seconds)
- Memory estimates (MB)
- Compression ratios
- Speedup comparisons
- Occupancy fractions
- Build time per node/cell metrics

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade open3d matplotlib numpy
```

**Open3D Visualization Errors:**
- The code disables Open3D visualization automatically
- If errors persist, ensure `OPEN3D_DISABLE_VISUALIZATION=1` is set

**Memory Issues with Large Datasets:**
- Reduce octree depth (e.g., depth=5 instead of 6)
- Increase voxel size (e.g., 0.2 instead of 0.1)
- Process datasets in chunks

**Plot Generation Errors:**
- Ensure `results/` directory exists
- Check matplotlib backend compatibility
- Verify write permissions

## Future Extensions

Possible improvements:

- **Performance**: Implement Octree in C++/Cython for speed
- **Functionality**: Add raycasting + free-space updates
- **Visualization**: Replace Matplotlib with Plotly for interactive 3D
- **Features**: Add color or intensity attributes from point cloud
- **Algorithms**: Use OctoMap-like Bayesian fusion
- **Optimization**: Implement node pruning and memory pooling
- **Analysis**: Add actual memory profiling (psutil) vs estimates

-----

## Author Notes

This project was built for 16-362 Robotics coursework.  
It focuses on clarity, reproducibility, and ease of experimentation rather than raw performance.

Feel free to modify any module — all code is isolated and modular.

```
```
