"""
Visualization utilities for document layout analysis.

This module provides functions to visualize various stages of the
Docstrum algorithm, including preprocessing results, components,
text lines, and final blocks.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.ndimage import convolve1d
from typing import List, Tuple

from segmentation import Component


def generate_distinct_colors(n_colors: int) -> np.ndarray:
    """
    Generate visually distinct colors for visualization.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        Array of RGB colors in 0-255 range
    """
    # Use multiple colormaps for variety
    base = plt.cm.tab20(np.linspace(0, 1, min(20, n_colors)))
    
    if n_colors > 20:
        base = np.vstack((base, plt.cm.Dark2(np.linspace(0, 1, min(8, n_colors-20)))))
    if n_colors > 28:
        base = np.vstack((base, plt.cm.Set1(np.linspace(0, 1, min(9, n_colors-28)))))
    
    # Adjust brightness for visibility
    min_brightness = 0.3
    max_brightness = 0.9
    
    for i in range(len(base)):
        # Calculate perceived brightness
        brightness = 0.299 * base[i,0] + 0.587 * base[i,1] + 0.114 * base[i,2]
        
        # Adjust too dark colors
        if brightness < min_brightness:
            scale = min_brightness / (brightness + 1e-6)
            base[i,:3] = np.minimum(base[i,:3] * scale, 1.0)
        
        # Adjust too bright colors
        if brightness > max_brightness:
            scale = max_brightness / (brightness + 1e-6)
            base[i,:3] = base[i,:3] * scale
    
    # Generate additional colors if needed
    while len(base) < n_colors:
        additional = base[:n_colors-len(base)]
        # Shift hue for variations
        hsv = matplotlib.colors.rgb_to_hsv(additional[:,:3])
        hsv[:,0] = (hsv[:,0] + 0.5) % 1.0
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        additional[:,:3] = rgb
        base = np.vstack((base, additional))
    
    return (base[:n_colors, :3] * 255).astype(int)


def save_figure_with_resolution(image: np.ndarray, output_path: str, dpi: int = 100):
    """
    Save a figure maintaining the original image resolution.
    
    Args:
        image: Image to save
        output_path: Output file path
        dpi: Dots per inch for saving
    """
    h, w = image.shape[:2]
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w/dpi, h/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(image, aspect='auto')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def visualize_preprocessing(image: np.ndarray, binary: np.ndarray, 
                          output_dir: str, filename: str):
    """
    Visualize and save the preprocessing result.
    
    Args:
        image: Original grayscale image
        binary: Binary image after preprocessing
        output_dir: Output directory
        filename: Base filename for output
    """
    output_path = os.path.join(output_dir, f'{filename}_preprocessing.png')
    save_figure_with_resolution(binary * 255, output_path)
    print(f"Saved preprocessing visualization to {output_path}")


def visualize_components(image: np.ndarray, components: List[Component], 
                       output_dir: str, filename: str):
    """
    Visualize detected connected components with bounding boxes.
    
    Draws bounding boxes and centroids for each component.
    Components are expected to have bbox in (x1, y1, x2, y2) format.
    
    Args:
        image: Original grayscale image
        components: List of Component objects with bbox as (x1, y1, x2, y2)
        output_dir: Output directory path
        filename: Base filename for output (without extension)
    """
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    colors = generate_distinct_colors(len(components))
    
    for idx, comp in enumerate(components):
        color = colors[idx % len(colors)].tolist()
        x1, y1, x2, y2 = comp.bbox
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 1)
        
        # Draw centroid
        cx, cy = map(int, comp.centroid)
        cv2.circle(vis_image, (cx, cy), 2, color, -1)
    
    output_path = os.path.join(output_dir, f'{filename}_components.png')
    save_figure_with_resolution(vis_image, output_path)
    print(f"Saved components visualization to {output_path}")


def visualize_neighbors(image: np.ndarray, components: List[Component],
                       neighbors_info: List[List[Tuple[int, float, float]]],
                       output_dir: str, filename: str):
    """
    Visualize k-nearest neighbor connections between components.
    
    Args:
        image: Original image
        components: List of Component objects
        neighbors_info: Neighbor information from Docstrum
        output_dir: Output directory
        filename: Base filename for output
    """
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    
    # Draw connections
    for i, component_neighbors in enumerate(neighbors_info):
        x1, y1 = map(int, components[i].centroid)
        
        for neighbor_idx, dist, angle in component_neighbors:
            x2, y2 = map(int, components[neighbor_idx].centroid)
            
            # Color based on angle
            color = plt.cm.hsv(angle / 180)[:3]
            color = tuple(int(c * 255) for c in color)
            
            cv2.line(vis_image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
    # Draw centroids on top
    for comp in components:
        cx, cy = map(int, comp.centroid)
        cv2.circle(vis_image, (cx, cy), 2, (255, 0, 0), -1)
    
    # Save main visualization
    output_path = os.path.join(output_dir, f'{filename}_neighbors.png')
    save_figure_with_resolution(vis_image, output_path)
    
    # Save version with colorbar
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(vis_image)
    ax.set_title('K-Nearest Neighbors Connections')
    ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(0, 180)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Angle (degrees)')
    
    colorbar_path = os.path.join(output_dir, f'{filename}_neighbors_with_colorbar.png')
    plt.savefig(colorbar_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved neighbors visualization to {output_path}")


def visualize_orientation_histogram(neighbors_info: List[List[Tuple[int, float, float]]],
                                  orientation: float, output_dir: str, 
                                  filename: str, smoothing_window: int):
    """
    Visualize histogram of neighbor angles and detected orientation.
    
    Args:
        neighbors_info: Neighbor information from Docstrum
        orientation: Detected orientation angle
        output_dir: Output directory
        filename: Base filename for output
        smoothing_window: Size of smoothing window used
    """
    # Collect angles
    angles = []
    for component_neighbors in neighbors_info:
        angles.extend([n[2] for n in component_neighbors])
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    hist, bins, _ = plt.hist(angles, bins=360, range=(0, 180), 
                           color='skyblue', alpha=0.7, edgecolor='black')
    
    # Apply smoothing for visualization
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed = convolve1d(hist, kernel, mode='wrap')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, smoothed, 'r-', linewidth=2, label='Smoothed')
    
    # Mark detected orientation
    plt.axvline(x=orientation+0.5, color='green', linestyle='--', linewidth=2,
                label=f'Detected Orientation: {orientation:.1f}°')
    
    plt.title('Histogram of Neighbor Angles')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, f'{filename}_orientation_histogram.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved orientation histogram to {output_path}")


def visualize_docstrum(components: List[Component],
                     neighbors_info: List[List[Tuple[int, float, float]]],
                     output_dir: str, filename: str):
    """
    Visualize the Docstrum plot (relative positions of neighbors).
    
    Args:
        components: List of Component objects
        neighbors_info: Neighbor information from Docstrum
        output_dir: Output directory
        filename: Base filename for output
    """
    # Extract centroids
    centroids = [comp.centroid for comp in components]
    
    # Collect relative positions
    plot_points = []
    for i in range(len(components)):
        centroid = centroids[i]
        
        for neighbor_idx, _, _ in neighbors_info[i]:
            neighbor = centroids[neighbor_idx]
            translated = np.array([neighbor[0] - centroid[0], 
                                 neighbor[1] - centroid[1]])
            plot_points.append(translated)
    
    plot_points = np.array(plot_points)
    
    # Create plot
    plt.figure(figsize=(10, 10))
    plt.scatter(plot_points[:, 0], plot_points[:, 1], 
               color='blue', marker='o', alpha=0.5, s=5)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Relative X')
    plt.ylabel('Relative Y')
    plt.title('Docstrum: Relative Positions of Neighbors to Each Centroid')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, f'{filename}_docstrum.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Docstrum plot to {output_path}")


def visualize_text_lines(image: np.ndarray, components: List[Component],
                       text_lines: List[List[int]], output_dir: str, filename: str):
    """
    Visualize detected text lines with different colors.
    
    Args:
        image: Original image
        components: List of Component objects
        text_lines: List of text lines
        output_dir: Output directory
        filename: Base filename for output
    """
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    colors = generate_distinct_colors(len(text_lines))
    
    for line_idx, line in enumerate(text_lines):
        color = colors[line_idx % len(colors)].tolist()
        
        # Draw components in line
        for comp_idx in line:
            x1, y1, x2, y2 = components[comp_idx].bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Connect components
        if len(line) > 1:
            for i in range(len(line) - 1):
                x1, y1 = components[line[i]].centroid
                x2, y2 = components[line[i + 1]].centroid
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.line(vis_image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
    output_path = os.path.join(output_dir, f'{filename}_text_lines.png')
    save_figure_with_resolution(vis_image, output_path)
    print(f"Saved text lines visualization to {output_path}")


def visualize_initial_blocks(image: np.ndarray, components: List[Component],
                           blocks: List[List[List[int]]], output_dir: str, filename: str):
    """
    Visualize initial text blocks before merging.
    
    Args:
        image: Original image
        components: List of Component objects
        blocks: List of blocks
        output_dir: Output directory
        filename: Base filename for output
    """
    visualize_blocks(image, components, blocks, output_dir, 
                    f"{filename}_initial_blocks", "Initial Text Blocks")


def visualize_final_blocks(image: np.ndarray, components: List[Component],
                         blocks: List[List[List[int]]], output_dir: str, 
                         filename: str, title: str = "Final Text Blocks"):
    """
    Visualize final text blocks after merging.
    
    Args:
        image: Original image
        components: List of Component objects
        blocks: List of blocks
        output_dir: Output directory
        filename: Base filename for output
        title: Title for the visualization
    """
    visualize_blocks(image, components, blocks, output_dir, 
                    f"{filename}_final_blocks", title)


def visualize_blocks(image: np.ndarray, components: List[Component],
                    blocks: List[List[List[int]]], output_dir: str,
                    filename: str, title: str = "Text Blocks"):
    """
    Generic function to visualize text blocks.
    
    Args:
        image: Original image
        components: List of Component objects
        blocks: List of blocks
        output_dir: Output directory
        filename: Output filename (without extension)
        title: Title for the visualization
    """
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    colors = generate_distinct_colors(len(blocks))
    
    for block_idx, block in enumerate(blocks):
        color = colors[block_idx % len(colors)].tolist()
        
        # Get all components in block
        block_components = [comp_idx for line in block for comp_idx in line]
        
        if not block_components:
            continue
        
        # Find block boundaries
        min_x = min(components[idx].bbox[0] for idx in block_components)
        min_y = min(components[idx].bbox[1] for idx in block_components)
        max_x = max(components[idx].bbox[2] for idx in block_components)
        max_y = max(components[idx].bbox[3] for idx in block_components)
        
        # Draw block rectangle
        padding = 3
        cv2.rectangle(vis_image, 
                    (min_x - padding, min_y - padding), 
                    (max_x + padding, max_y + padding), 
                    color, 2)
    
    # Save using OpenCV for better quality
    output_path = os.path.join(output_dir, f'{filename}.png')
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"Saved {title.lower()} visualization to {output_path}")