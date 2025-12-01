Here is the corrected, pure Markdown version.

**Changes made:**

1.  **Fixed broken code blocks:** In your version, the opening backticks ` ``` ` were separated from the commands. I combined them so they render correctly on GitHub/VS Code.
2.  **Added Syntax Highlighting:** I added `bash` to terminal commands and `text` to the file tree so GitHub colors them appropriately.
3.  **Standardized Spacing:** Ensured consistent spacing between headers and lists.

You can copy the block below directly into your `README.md` file.

````markdown
# Octree-Based 3D Occupancy Mapping
### Carnegie Mellon University — 16-362 Mobile Robot Programming Lab

This project implements a full 3D occupancy Octree mapping pipeline using Python and compares it to a dense voxel grid baseline (Open3D).  
The project includes:

- A Python-based Octree with probabilistic occupancy (log-odds update)
- Point cloud loading and preprocessing
- Dense voxel grid mapping baseline (Open3D)
- Mac-safe visualizations using Matplotlib (no Open3D GUI needed)
- Automated benchmarking across multiple Octree depths
- Runtime and memory comparison plots

This README explains the directory structure, usage, and purpose of each file.

---

## Project Structure

```text
octree_mapping/
│
├── data/                        # Input point cloud data (.pcd)
│     └── (auto-downloaded by Open3D)
│
├── results/                     # Output plots and benchmark results
│     ├── runtime_vs_depth.png
│     ├── memory_vs_depth.png
│     └── octree_resolution.png
│
├── src/
│     ├── main.py                # Runs single Octree vs Dense Grid test
│     ├── benchmark.py           # Multi-depth benchmark + saves plots
│     ├── mapping.py             # Loads point clouds + builds Octree
│     ├── octree.py              # Octree implementation + insertion
│     ├── compare.py             # Dense voxel grid baseline (Open3D)
│     ├── visualize.py           # Matplotlib visualization for Octree
│     ├── metrics.py             # Metrics: node counts, bbox, statistics
│     └── plots.py               # Plot generation utilities
│
├── venv/                        # Python virtual environment
│
└── README.md                    # Documentation (this file)
````

-----

## Installation & Setup

### 1\. Create and activate a Python 3.10 virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 2\. Install required packages

```bash
pip install open3d==0.17 matplotlib numpy
```

*(Open3D GUI is disabled — only Matplotlib visualizations are used.)*

-----

## Running the Project

### Run a single test (one Octree + one dense voxel grid)

```bash
python src/main.py
```

This will:

  - Load `fragment.pcd` from Open3D’s dataset
  - Build a dense voxel grid
  - Build an Octree (depth = 6 by default)
  - Print metrics for both
  - Display Matplotlib 3D scatter visualizations

-----

## Running Benchmarks (Generates All Plots)

To benchmark performance across multiple Octree depths:

```bash
python src/benchmark.py
```

This script automatically:

  - Tests depths `[3, 4, 5, 6, 7]`
  - Measures runtime, node count, memory usage
  - Compares against a dense voxel grid baseline
  - Saves three plots into the `results/` folder:
      - `runtime_vs_depth.png`
      - `memory_vs_depth.png`
      - `octree_resolution.png`

These plots are used for your final report.

-----

## Key Files Explained

### `octree.py`

Defines:

  - `OctreeNode`
  - Probabilistic log-odds updates
  - Node subdivision rules
  - Point insertion logic

This is the core Octree implementation.

### `mapping.py`

Handles:

  - Loading `.pcd` point clouds with Open3D
  - Building the Octree using multiple insertions

### `compare.py`

Contains:

  - Dense voxel grid creation (`VoxelGrid.create_from_point_cloud`)
  - Matplotlib-safe voxel visualization

### `visualize.py`

Plots Octree occupied nodes using Matplotlib:

  - Each occupied leaf visualized as a colored scatter point
  - No Open3D visualizer required (avoids macOS GLFW crash)

### `metrics.py`

Provides:

  - Node count (all nodes)
  - Occupied leaf count
  - Bounding box computation
  - Pretty metric printing for console output

### `plots.py`

Generates:

  - Runtime vs depth
  - Memory usage vs depth
  - Octree node growth vs depth

Saves `.png` figures for your report.

-----

## Notes for Developers / Teammates

  - The Octree is intentionally pure Python (slow but educational).
  - The dense voxel grid uses optimized C++ (Open3D), so runtime comparisons are not direct competition.
  - Visualization is Matplotlib-only due to macOS GUI issues with Open3D.
  - All benchmarking plots are reproducible via `benchmark.py`.

-----

## Future Extensions

Possible improvements:

  - Implement Octree in C++ for speed
  - Add raycasting + free-space updates
  - Replace Matplotlib with modern 3D plotting (Plotly)
  - Add color or intensity attributes from point cloud
  - Use OctoMap-like Bayesian fusion

-----

## Author Notes

This project was built for 16-362 Robotics coursework.  
It focuses on clarity, reproducibility, and ease of experimentation rather than raw performance.

Feel free to modify any module — all code is isolated and modular.

```
```