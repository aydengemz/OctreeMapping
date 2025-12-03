import matplotlib.pyplot as plt
import numpy as np

def plot_runtime_comparison(dense_runtime, octree_runtime, dataset_name="", filename="results/runtime_comparison.png"):
    """
    Create a simple bar chart comparing dense voxel grid vs octree runtime.
    Perfect for showing Python implementation comparison.
    """
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.2], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    methods = ['Dense Voxel\nGrid (Python)', 'Octree\n(Python)']
    runtimes = [dense_runtime, octree_runtime]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(methods, runtimes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Runtime Comparison{f" - {dataset_name}" if dataset_name else ""}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{runtime:.4f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    speedup = dense_runtime / octree_runtime if octree_runtime > 0 else float('inf')
    faster_method = "Octree" if speedup > 1 else "Dense Grid"
    
    ax2.barh([0], [speedup if speedup <= 10 else 10], color='green' if speedup > 1 else 'orange', 
             alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Speedup Ratio', fontsize=12, fontweight='bold')
    ax2.set_title(f'Speedup: {faster_method} is\n{speedup:.2f}x faster', 
                 fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Break-even')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(fontsize=9)
    
    if speedup <= 10:
        ax2.text(speedup/2, 0, f'{speedup:.2f}x', 
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    else:
        ax2.text(5, 0, f'{speedup:.2f}x\n(truncated)', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax3.axis('off')
    explanation_text = "Why Dense Grid is Faster\n(for smaller/denser datasets):\n\n"
    explanation_text += "✓ O(1) dictionary insertion\n"
    explanation_text += "✓ No tree traversal overhead\n"
    explanation_text += "✓ No node subdivision logic\n"
    explanation_text += "✓ Direct hash-based lookup\n\n"
    explanation_text += "Octree overhead:\n"
    explanation_text += "• Depth-level tree traversal\n"
    explanation_text += "• Subdivision checks per point\n"
    explanation_text += "• More complex data structure"
    
    if speedup < 1:
        explanation_text = "Why Octree is Faster\n(for larger/sparser datasets):\n\n"
        explanation_text += "✓ Adapts to sparse occupancy\n"
        explanation_text += "✓ Fewer nodes in sparse regions\n"
        explanation_text += "✓ Better memory efficiency\n\n"
        explanation_text += "Dense Grid overhead:\n"
        explanation_text += "• Processes all points\n"
        explanation_text += "• Less adaptive to sparsity"
    
    ax3.text(0.1, 0.5, explanation_text, fontsize=10, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Python Implementation Runtime Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Runtime comparison plot saved to {filename}")


def plot_runtime(depths, runtimes_octree, runtimes_dense, filename="results/runtime_vs_depth.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(depths, runtimes_octree, marker='o', label="Octree Runtime", linewidth=2, markersize=8)
    plt.plot(depths, runtimes_dense, marker='s', label="Dense Voxel Grid Runtime", linewidth=2, markersize=8, linestyle='--')
    plt.xlabel("Octree Depth", fontsize=12)
    plt.ylabel("Runtime (sec)", fontsize=12)
    plt.title("Runtime Comparison: Octree vs Dense Voxel Grid", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_memory(depths, octree_leaves, dense_voxels, filename="results/memory_vs_depth.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(depths, octree_leaves, marker='o', label="Octree Occupied Leaves", linewidth=2, markersize=8)
    plt.plot(depths, dense_voxels, marker='s', label="Dense Voxel Count", linewidth=2, markersize=8, linestyle='--')
    plt.xlabel("Octree Depth", fontsize=12)
    plt.ylabel("Number of Cells", fontsize=12)
    plt.title("Memory Usage: Occupied Cells Comparison", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_resolution(depths, octree_nodes, filename="results/octree_resolution.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(depths, octree_nodes, marker='o', label="Octree Total Nodes", linewidth=2, markersize=8, color='green')
    plt.xlabel("Octree Depth", fontsize=12)
    plt.ylabel("Total Nodes Allocated", fontsize=12)
    plt.title("Octree Resolution Adaptability", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_build_time_per_cell(depths, build_time_octree, build_time_dense):
    """Plot build time per node/cell (microseconds)."""
    plt.figure(figsize=(10, 6))
    plt.plot(depths, build_time_octree, marker='o', label="Octree (μs/node, includes all nodes)", linewidth=2, markersize=8)
    plt.plot(depths, build_time_dense, marker='s', label="Dense Grid (μs/voxel)", linewidth=2, markersize=8, linestyle='--')
    plt.xlabel("Octree Depth", fontsize=12)
    plt.ylabel("Build Time per Node/Cell (microseconds)", fontsize=12)
    plt.title("Build Time Efficiency: Time per Node/Cell", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/build_time_per_cell.png", dpi=150)
    plt.close()

def plot_memory_per_cell(depths, memory_octree, memory_dense):
    """Plot estimated memory usage per node/cell (bytes)."""
    plt.figure(figsize=(10, 6))
    plt.plot(depths, memory_octree, marker='o', label="Octree (bytes/node, estimated, includes all nodes)", linewidth=2, markersize=8)
    plt.plot(depths, memory_dense, marker='s', label="Dense Grid (bytes/voxel, estimated)", linewidth=2, markersize=8, linestyle='--')
    plt.xlabel("Octree Depth", fontsize=12)
    plt.ylabel("Estimated Memory per Node/Cell (bytes)", fontsize=12)
    plt.title("Memory Efficiency: Estimated Bytes per Node/Cell", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/memory_per_cell.png", dpi=150)
    plt.close()

def plot_efficiency_comparison(depths, octree_memory, dense_memory, octree_leaves, dense_voxels, 
                               filename="results/efficiency_comparison.png"):
    """Plot comprehensive efficiency comparison."""
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    octree_mem_mb = [m / (1024*1024) for m in octree_memory]
    dense_mem_mb = [m / (1024*1024) for m in dense_memory]
    ax1.plot(depths, octree_mem_mb, marker='o', label="Octree", linewidth=2, markersize=8)
    ax1.plot(depths, dense_mem_mb, marker='s', label="Dense Grid", linewidth=2, markersize=8, linestyle='--')
    ax1.set_xlabel("Octree Depth", fontsize=11)
    ax1.set_ylabel("Estimated Memory (MB)", fontsize=11)
    ax1.set_title("Total Estimated Memory Usage", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    compression_ratios = [dense_voxels[0] / occ for occ in octree_leaves]
    ax2.plot(depths, compression_ratios, marker='o', label="Compression Ratio", linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel("Octree Depth", fontsize=11)
    ax2.set_ylabel("Compression Ratio", fontsize=11)
    ax2.set_title("Space Efficiency (Dense/Octree cells)", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='r', linestyle=':', alpha=0.5, label='Break-even')
    
    ax3 = axes[1, 0]
    memory_ratios = [dense_memory[0] / mem for mem in octree_memory]
    ax3.plot(depths, memory_ratios, marker='o', label="Memory Ratio", linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel("Octree Depth", fontsize=11)
    ax3.set_ylabel("Memory Ratio (Dense/Octree)", fontsize=11)
    ax3.set_title("Memory Efficiency Gain", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='r', linestyle=':', alpha=0.5, label='Break-even')
    
    ax4 = axes[1, 1]
    ax4.plot(depths, octree_leaves, marker='o', label="Octree Occupied Leaves", linewidth=2, markersize=8)
    ax4.plot(depths, dense_voxels, marker='s', label="Dense Voxels", linewidth=2, markersize=8, linestyle='--')
    ax4.set_xlabel("Octree Depth", fontsize=11)
    ax4.set_ylabel("Number of Cells (log scale)", fontsize=11)
    ax4.set_title("Occupied Cell Count Comparison", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.suptitle("Comprehensive Efficiency Comparison: Octree vs Dense Voxel Grid", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_dataset_comparison(real_data, sparse_data):
    """Compare real dataset vs synthetic sparse world side-by-side."""
    _, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(real_data['depths'], real_data['runtimes_octree'], marker='o', 
             label="Real - Octree", linewidth=2, markersize=8, color='blue')
    ax1.plot(real_data['depths'], real_data['runtimes_dense'], marker='s', 
             label="Real - Dense", linewidth=2, markersize=8, linestyle='--', color='lightblue')
    ax1.plot(sparse_data['depths'], sparse_data['runtimes_octree'], marker='o', 
             label="Sparse - Octree", linewidth=2, markersize=8, color='red')
    ax1.plot(sparse_data['depths'], sparse_data['runtimes_dense'], marker='s', 
             label="Sparse - Dense", linewidth=2, markersize=8, linestyle='--', color='pink')
    ax1.set_xlabel("Octree Depth", fontsize=11)
    ax1.set_ylabel("Runtime (sec)", fontsize=11)
    ax1.set_title("Runtime Comparison", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    real_mem = [m / (1024*1024) for m in real_data['octree_memory']]
    sparse_mem = [m / (1024*1024) for m in sparse_data['octree_memory']]
    ax2.plot(real_data['depths'], real_mem, marker='o', 
             label="Real - Octree", linewidth=2, markersize=8, color='blue')
    ax2.plot(real_data['depths'], [real_data['dense_mem']/(1024*1024)]*len(real_data['depths']), 
             marker='s', label="Real - Dense", linewidth=2, markersize=8, linestyle='--', color='lightblue')
    ax2.plot(sparse_data['depths'], sparse_mem, marker='o', 
             label="Sparse - Octree", linewidth=2, markersize=8, color='red')
    ax2.plot(sparse_data['depths'], [sparse_data['dense_mem']/(1024*1024)]*len(sparse_data['depths']), 
             marker='s', label="Sparse - Dense", linewidth=2, markersize=8, linestyle='--', color='pink')
    ax2.set_xlabel("Octree Depth", fontsize=11)
    ax2.set_ylabel("Estimated Memory (MB)", fontsize=11)
    ax2.set_title("Memory Usage Comparison", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax3 = axes[0, 2]
    ax3.plot(real_data['depths'], real_data['octree_nodes'], marker='o', 
             label="Real - Total Nodes", linewidth=2, markersize=8, color='blue')
    ax3.plot(real_data['depths'], real_data['octree_leaves'], marker='^', 
             label="Real - Occupied Leaves", linewidth=2, markersize=8, color='lightblue')
    ax3.plot(sparse_data['depths'], sparse_data['octree_nodes'], marker='o', 
             label="Sparse - Total Nodes", linewidth=2, markersize=8, color='red')
    ax3.plot(sparse_data['depths'], sparse_data['octree_leaves'], marker='^', 
             label="Sparse - Occupied Leaves", linewidth=2, markersize=8, color='pink')
    ax3.set_xlabel("Octree Depth", fontsize=11)
    ax3.set_ylabel("Number of Nodes", fontsize=11)
    ax3.set_title("Octree Node Growth", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4 = axes[1, 0]
    real_compression = [real_data['theoretical_dense_cells'][0] / n for n in real_data['octree_nodes']]
    sparse_compression = [sparse_data['theoretical_dense_cells'][0] / n for n in sparse_data['octree_nodes']]
    ax4.plot(real_data['depths'], real_compression, marker='o', 
             label="Real Dataset", linewidth=2, markersize=8, color='blue')
    ax4.plot(sparse_data['depths'], sparse_compression, marker='s', 
             label="Sparse World", linewidth=2, markersize=8, color='red')
    ax4.set_xlabel("Octree Depth", fontsize=11)
    ax4.set_ylabel("Compression Ratio", fontsize=11)
    ax4.set_title("Compression vs Full Dense Grid", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    ax5 = axes[1, 1]
    real_occ = [real_data['dense_voxel_count'] / real_data['theoretical_dense_cells'][0]] * len(real_data['depths'])
    sparse_occ = [sparse_data['dense_voxel_count'] / sparse_data['theoretical_dense_cells'][0]] * len(sparse_data['depths'])
    ax5.bar(['Real Dataset', 'Sparse World'], 
            [real_occ[0]*100, sparse_occ[0]*100],
            color=['blue', 'red'], alpha=0.7)
    ax5.set_ylabel("Occupancy Fraction (%)", fontsize=11)
    ax5.set_title("World Occupancy (Occupied/Total Cells)", fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_yscale('log')
    
    ax6 = axes[1, 2]
    categories = ['Real Dataset', 'Sparse World']
    theoretical = [real_data['theoretical_dense_cells'][0], sparse_data['theoretical_dense_cells'][0]]
    occupied = [real_data['dense_voxel_count'], sparse_data['dense_voxel_count']]
    x = np.arange(len(categories))
    width = 0.35
    ax6.bar(x - width/2, [t/(1e6) for t in theoretical], width, 
            label='Theoretical Full Grid (M cells)', color='orange', alpha=0.7)
    ax6.bar(x + width/2, [o/(1e6) for o in occupied], width, 
            label='Actually Occupied (M cells)', color='green', alpha=0.7)
    ax6.set_ylabel("Number of Cells (Millions)", fontsize=11)
    ax6.set_title("Theoretical vs Actual Cell Count", fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    plt.suptitle("Dataset Comparison: Real vs Synthetic Sparse World", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig("results/dataset_comparison.png", dpi=150)
    plt.close()


def plot_theoretical_dense_comparison(real_data, sparse_data):
    """Show how octrees compare to theoretical full dense grids."""
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    real_theoretical_mem = real_data['theoretical_dense_cells'][0] * 50 / (1024*1024)
    sparse_theoretical_mem = sparse_data['theoretical_dense_cells'][0] * 50 / (1024*1024)
    real_octree_mem = [m / (1024*1024) for m in real_data['octree_memory']]
    sparse_octree_mem = [m / (1024*1024) for m in sparse_data['octree_memory']]
    
    ax1.plot(real_data['depths'], [real_theoretical_mem]*len(real_data['depths']), 
             marker='s', label="Real - Theoretical Dense", linewidth=2, markersize=8, 
             linestyle='--', color='lightblue')
    ax1.plot(real_data['depths'], real_octree_mem, marker='o', 
             label="Real - Octree", linewidth=2, markersize=8, color='blue')
    ax1.plot(sparse_data['depths'], [sparse_theoretical_mem]*len(sparse_data['depths']), 
             marker='s', label="Sparse - Theoretical Dense", linewidth=2, markersize=8, 
             linestyle='--', color='pink')
    ax1.plot(sparse_data['depths'], sparse_octree_mem, marker='o', 
             label="Sparse - Octree", linewidth=2, markersize=8, color='red')
    ax1.set_xlabel("Octree Depth", fontsize=11)
    ax1.set_ylabel("Estimated Memory (MB)", fontsize=11)
    ax1.set_title("Theoretical Dense Grid vs Octree Memory", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = axes[0, 1]
    real_savings = [real_theoretical_mem / m for m in real_octree_mem]
    sparse_savings = [sparse_theoretical_mem / m for m in sparse_octree_mem]
    ax2.plot(real_data['depths'], real_savings, marker='o', 
             label="Real Dataset", linewidth=2, markersize=8, color='blue')
    ax2.plot(sparse_data['depths'], sparse_savings, marker='s', 
             label="Sparse World", linewidth=2, markersize=8, color='red')
    ax2.set_xlabel("Octree Depth", fontsize=11)
    ax2.set_ylabel("Memory Savings Ratio", fontsize=11)
    ax2.set_title("Octree Memory Savings vs Full Dense Grid", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    
    ax3 = axes[1, 0]
    ax3.plot(real_data['depths'], [real_data['theoretical_dense_cells'][0]/(1e6)]*len(real_data['depths']), 
             marker='s', label="Real - Theoretical", linewidth=2, markersize=8, 
             linestyle='--', color='lightblue')
    ax3.plot(real_data['depths'], [n/(1e6) for n in real_data['octree_nodes']], 
             marker='o', label="Real - Octree Nodes", linewidth=2, markersize=8, color='blue')
    ax3.plot(sparse_data['depths'], [sparse_data['theoretical_dense_cells'][0]/(1e6)]*len(sparse_data['depths']), 
             marker='s', label="Sparse - Theoretical", linewidth=2, markersize=8, 
             linestyle='--', color='pink')
    ax3.plot(sparse_data['depths'], [n/(1e6) for n in sparse_data['octree_nodes']], 
             marker='o', label="Sparse - Octree Nodes", linewidth=2, markersize=8, color='red')
    ax3.set_xlabel("Octree Depth", fontsize=11)
    ax3.set_ylabel("Number of Cells (Millions)", fontsize=11)
    ax3.set_title("Cell Count: Theoretical Dense vs Octree", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4 = axes[1, 1]
    real_comp = [real_data['theoretical_dense_cells'][0] / n for n in real_data['octree_nodes']]
    sparse_comp = [sparse_data['theoretical_dense_cells'][0] / n for n in sparse_data['octree_nodes']]
    ax4.plot(real_data['depths'], real_comp, marker='o', 
             label="Real Dataset", linewidth=2, markersize=8, color='blue')
    ax4.plot(sparse_data['depths'], sparse_comp, marker='s', 
             label="Sparse World", linewidth=2, markersize=8, color='red')
    ax4.set_xlabel("Octree Depth", fontsize=11)
    ax4.set_ylabel("Compression Factor", fontsize=11)
    ax4.set_title("Octree Compression vs Full Dense Grid", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    ax4.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    
    plt.suptitle("Theoretical Full Dense Grid vs Octree: Why Octrees Excel on Sparse Worlds", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig("results/theoretical_dense_comparison.png", dpi=150)
    plt.close()
