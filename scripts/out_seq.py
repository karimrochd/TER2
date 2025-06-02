# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from numba import njit
from scipy.ndimage import uniform_filter1d
from scipy.spatial import KDTree
from skimage.filters import threshold_otsu

import vpv

# %%
img = iio.read('test1.png').mean(axis=2)
iio.display(img)


# %%
def binarize(img):
    """Binarize the image with Otsu thresholding; foreground (minority) is 1."""
    # Apply Otsu's thresholding
    thresh = threshold_otsu(img)
    img_binary = (img < thresh)
    
    # Invert if needed we want text = foreground = 1
    if np.mean(img_binary) > 0.5:
        img_binary = ~img_binary
    
    return img_binary


# %%
img_binary = binarize(img)
plt.imshow(img_binary)


# %%
@njit
def compute_kfill(img_binary, k, max_iterations):
    h, w = img_binary.shape
    img_filtered = img_binary.astype(np.int32)
    iteration = 0
    changes_made = True

    while changes_made and iteration < max_iterations:
        changes_made = False
        iteration += 1

        for fill_value in (1,):  # ON-fill and OFF-fill
            img_temp = img_filtered.copy()

            for y in range(k//2, h-k//2):
                for x in range(k//2, w-k//2):
                    window = img_filtered[y-k//2:y+k//2+1, x-k//2:x+k//2+1]
                    core = window[1:-1, 1:-1]

                    # # Only proceed if not all core values are fill_value
                    # # np.any() avoided for numba
                    # skip_core = True
                    # for cy in range(core.shape[0]):
                    #     for cx in range(core.shape[1]):
                    #         if core[cy, cx] != fill_value:
                    #             skip_core = False
                    #             break
                    #     if not skip_core:
                    #         break
                    # if skip_core:
                    #     continue
                    # Only proceed if all core values are opposite of fill_value
                    # np.any() avoided for numba
                    skip_core = False
                    for cy in range(core.shape[0]):
                        for cx in range(core.shape[1]):
                            if core[cy, cx] == fill_value:
                                skip_core = True
                                break
                        if skip_core:
                            break
                    if skip_core:
                        continue

                    # Extract neighborhood (perimeter of window)
                    # np.concatenate() avoided for numba
                    nbhd_len = 4*(k-1)
                    nbhd = np.empty(nbhd_len, dtype=np.int32)
                    for i in range(k-1):
                        nbhd[i]         = window[0, i]
                        nbhd[k-1+i]     = window[i, -1]
                        nbhd[2*(k-1)+i] = window[-1, k-1-i]
                        nbhd[3*(k-1)+i] = window[k-1-i, 0]

                    n = np.sum(nbhd == fill_value)

                    # Only proceed if c is 1
                    # (c is the number of non-looping connected chains
                    # of pixels with fill_value in the cornerless neighborhood,
                    # plus number of isolated corner pixels with fill_value;
                    # equivalently, number of non-looping connected chains of
                    # pixels with fill_value in the neighborhood, where
                    # connectedness is defined by 8-connectivity)
                    tra = nbhd - np.roll(nbhd, 1)
                    c = np.sum(tra == 1)
                    c -= np.sum((tra[::k-1] == 2*fill_value-1) &
                                (tra[1::k-1] == -2*fill_value+1))
                    if c != 1:
                        continue

                    # Compute r (number of corner pixels that are fill_value)
                    r = np.sum(nbhd[::k-1] == fill_value)

                    # Apply kFill condition (note that c==1 is already ensured)
                    if n > 3*k-4 or (n == 3*k-4 and r == 2):
                        for cy in range(core.shape[0]):
                            for cx in range(core.shape[1]):
                                img_temp[y-k//2+1+cy, x-k//2+1+cx] = fill_value
                        changes_made = True

            img_filtered = img_temp

    return img_filtered, iteration

def kfill(img_binary, k=5, max_iterations=10, verbose=False):
    if k % 2 == 0:
        k += 1
    img_out, it = compute_kfill(img_binary, k, max_iterations)

    if verbose:
        print(f"Completed {it} iterations.")

    return img_out.astype(bool)



# %%
def kfill2(img_binary, k=5, max_iterations=10, verbose=False):
    """
    Implement the kFill filter for noise reduction in binary document images.
    This is the Will filter mentioned in the paper.
    
    Args:
        img_binary (numpy.ndarray): Binary image (1 for foreground, 0 for background).
        k (int): Window size parameter (must be odd).
        max_iterations (int): Maximum number of iterations to perform.
        
    Returns:
        numpy.ndarray: Filtered binary image.
    """
    # Ensure k is odd
    k = k if k % 2 == 1 else k + 1
    
    # Create a copy of the image
    img_filtered = img_binary.astype(np.int32)
    
    iteration = 0
    changes_made = True
    
    # Continue until no changes or max iterations reached
    while changes_made and iteration < max_iterations:
        changes_made = False
        iteration += 1
        
        # Perform ON-fill and OFF-fill sub-iterations
        for fill_value in [1]:  # 1 for ON-fill, 0 for OFF-fill
            h, w = img_filtered.shape
            
            # Create a copy to store changes for this sub-iteration
            img_temp = img_filtered.copy()
            
            # Process each pixel
            for y in range(k//2, h-k//2):
                for x in range(k//2, w-k//2):
                    # Extract window
                    window = img_filtered[y-k//2 :y+k//2+1, x-k//2:x+k//2+1]
                    
                    # Define core and neighborhood
                    core = window[1:-1, 1:-1]
                    
                    # Only proceed if all core values are opposite of fill_value
                    if np.any(core == fill_value):
                        continue
                    
                    # Extract neighborhood (perimeter of window)
                    nbhd = np.concatenate([
                        window[0, :-1],             # Top row
                        window[:-1, -1],            # Right column
                        window[-1, :0:-1],          # Bottom row
                        window[:0:-1, 0],           # Left column
                    ])
                    
                    # Compute n (number of fill_value pixels in neighborhood)
                    n = np.sum(nbhd == fill_value)
                    
                    # Only proceed if c is 1
                    # (c is the number of non-looping connected chains
                    # of pixels with fill_value in the cornerless neighborhood,
                    # plus number of isolated corner pixels with fill_value;
                    # equivalently, number of non-looping connected chains of
                    # pixels with fill_value in the neighborhood, where
                    # connectedness is defined by 8-connectivity)
                    tra = nbhd - np.roll(nbhd, 1) # transition from pixel i-1 to pixel i
                    c = np.sum(tra==1) - np.sum((tra[::k-1]==2*fill_value-1) &
                                                (tra[1::k-1]==-2*fill_value+1))
                    if c != 1:
                        continue
                    
                    # Compute r (number of corner pixels that are fill_value)
                    r = np.sum(nbhd[::k-1] == fill_value)
                    
                    # Apply kFill condition (note that c==1 is already ensured)
                    if n > 3*k-4 or (n == 3*k-4 and r == 2):
                        # Fill the core
                        img_temp[y-k//2+1:y+k//2, x-k//2+1:x+k//2] = fill_value
                        changes_made = True
            
            # Update img_filtered with the results of this sub-iteration
            img_filtered = img_temp.copy()
    
    if verbose:
        print(f"Completed {iteration} iterations.")
    
    return img_filtered.astype(np.bool_)


# %%
img_filtered = kfill(img_binary, k=5, verbose=True)
plt.imshow(img_filtered)
plt.show()


# %%
@dataclass
class Component:
    """Class to store connected component information"""
    bbox: np.ndarray  # x, y, w, h, dtype=np.int32
    centroid: np.ndarray   # x, y, dtype=np.float64
    area: int


def find_connected_components(img_binary: np.ndarray) -> List[Component]:
        """
        Find connected components in binary image using contours as described in the paper.
        The paper mentions using "thin line code" (TLC), but we'll use OpenCV contours
        which provide the necessary features.
        
        Args:
            binary: Binary image
            
        Returns:
            List of Component objects
        """
        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            img_binary.astype(np.uint8), connectivity=8)
        
        return [Component(bbox=stat[:4],
                           centroid=centroid,
                           area=stat[4].item())
                for stat, centroid in zip(stats[1:], centroids[1:])]


# %%
comps_raw = find_connected_components(img_filtered)
len(comps_raw)


# %%
def size_filtering(components: List[Component], verbose=False) -> List[Component]:
    """
    Filter components based on size histogram as described in the paper.
    
    Args:
        components: List of components
        
    Returns:
        Filtered list of components
    """
    if not components:
        return []
        
    # Compute size for each component (square root of bounding box area)
    sizes = [np.sqrt(comp.bbox[2] * comp.bbox[3]) for comp in components]
    
    # TODO: carefully check histogram binning and peak detection

    # Create histogram of sizes
    hist, bins = np.histogram(sizes, bins='auto')
    
    # Find the peak for predominant font size
    peak_idx = np.argmax(hist)
    peak_size = (bins[peak_idx] + bins[peak_idx+1]) / 2

    if verbose:
        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge', edgecolor='black', alpha=0.7)
        plt.axvline(peak_size, color='red', linestyle='--', label='Peak Size')
        plt.xscale('log')
        plt.xlabel('Component Size (sqrt(area))')
        plt.ylabel('Frequency')
        plt.title('Size Histogram of Components')
        plt.legend()
        plt.show()
    
    # Filter components - keep those within a reasonable range of peak size
    min_size = 3
    max_size = 3 * peak_size

    filtered_components = [comp
                           for comp, size in zip(components, sizes)
                           if min_size <= size <= max_size]
    
    return filtered_components


# %%
comps = size_filtering(comps_raw, verbose=True)
len(comps)


# %%
def visualize_components(img_binary: np.ndarray, components: List[Component]):
    """
    Visualize and save detected connected components with bounding boxes and centroids
    
    Args:
        image: Original grayscale image
        components: List of components
        output_dir: Output directory path
        filename: Base filename for the output
    """
    # Create RGB visualization image
    img_vis = np.stack([(~img_binary).astype(np.uint8)*255] * 3, axis=-1)
    
    # Randomly generate distinct colors
    rng = np.random.default_rng(0)
    colors = plt.cm.rainbow(rng.random(len(components)))
    colors = np.rint((colors[:, :3] * 255)).astype(int)
    
    # Draw components
    for comp, color in zip(components, colors):
        # Draw centroid
        cx, cy = np.rint(comp.centroid).astype(int)
        img_vis[cy, cx] = color
        
        # Draw bounding box
        x, y, w, h = comp.bbox
        color = tuple(map(int, color))
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), color, 2)
        

    return img_vis
    
    # # Get original image dimensions
    # h, w = img_binary.shape[:2]
    
    # # Create a figure with same aspect ratio
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(w/100, h/100)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    
    # # Display the visualization
    # ax.imshow(vis_image)
    
    # # Save the figure with the same resolution as the input image
    # os.makedirs(output_dir, exist_ok=True)
    # # output_path = os.path.join(output_dir, f'{filename}_components.png')
    # # plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    # # plt.close(fig)
    # plt.show()

# %%
img_comps = visualize_components(img_binary, comps)
plt.imshow(img_comps)
vpv(img_comps)


# %%
def find_nearest_neighbors(components: List[Component], k=5) -> List[List[Tuple[int, float, float]]]:
    """
    Find k nearest neighbors for each component
    
    Args:
        components: List of components
        
    Returns:
        # List of lists containing (neighbor_idx, distance, angle) tuples for each component
        List of lists containing vectors to k nearest neighbors
    """
    if len(components) < k+1:
        raise ValueError(f"Not enough components ({len(components)}) for k={k} nearest neighbors")
        
    # Extract centroids
    points = np.array([c.centroid for c in components])
    
    # Build KD-tree for efficient nearest neighbor search
    tree = KDTree(points)
    
    # Find k nearest neighbors (first one is the point itself)
    distances, indices = tree.query(points, k=k+1) # shapes (n, k+1) and (n, k+1)

    # Get vectors to nearest neighbors
    vectors = points[indices[:, 1:]] - points[:, None, :] # shape (n, k, 2)

    return indices[:, 1:], vectors


# %%
idxs_ngbr, vects_ngbr = find_nearest_neighbors(comps, k=5)


# %%
def visualize_neighbors(img_binary: np.ndarray,
                        components: List[Component], 
                        vectors: List[List[Tuple[int, float, float]]]):
    """
    Visualize and save k-nearest neighbors connections between components
    
    Args:
        image: Original grayscale image
        components: List of components
        neighbors_info: List of neighbor information
        output_dir: Output directory path
        filename: Base filename for the output
    """
    # Create RGB visualization image
    img_vis = np.stack([(~img_binary).astype(np.uint8)*255] * 3, axis=-1)
    
    # Draw connections between components
    for comp, vectors_neighbors in zip(components, vectors):
        x1, y1 = map(int, np.rint(comp.centroid))
        
        for vector in vectors_neighbors:
            x2, y2 = map(int, np.rint(vector+comp.centroid))
            
            # Color based on angle (cyclic color map)
            angle = np.arctan2(vector[1], vector[0])
            color = tuple(int(np.rint(c*255))
                          for c in plt.cm.hsv((angle/np.pi)%1)[:3])
            
            # Draw line connecting components
            cv2.line(img_vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
        # Draw component centroids on top
        cv2.circle(img_vis, (x1, y1), 2, (255, 0, 0), -1)
    
    return img_vis


# %%
img_neighbors = visualize_neighbors(img_binary, comps, vects_ngbr)
plt.imshow(img_neighbors)
vpv(img_neighbors)


# %%
def visualize_docstrum(vectors):
    """
    Visualize and save the docstrum plot (relative positions of neighbors to each centroid)
    
    Args:
        components: List of components
        neighbors_info: List of neighbor information
        output_dir: Output directory path
        filename: Base filename for the output
    """
    # Plot the neighbors as blue circles
    plt.scatter(vectors[..., 0], vectors[..., 1],
                color='blue', marker='o', alpha=0.5, s=5)
    # Symmetrize the plot
    plt.scatter(-vectors[..., 0], -vectors[..., 1],
                color='blue', marker='o', alpha=0.5, s=5)
    
    # Set axes properties
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Relative X')
    plt.ylabel('Relative Y')
    plt.title('Docstrum: Relative Positions of Neighbors to Each Centroid')
    plt.grid(True)
    
    plt.show()


# %%
visualize_docstrum(vects_ngbr)


# %%
def estimate_orientation(vectors, show=False) -> float:
    """
    Estimate document orientation from neighbor angles as described in the paper.
    
    Args:
        neighbors_info: List of neighbor information
        smoothing_window: Size of the smoothing window (default 25)
        
    Returns:
        Estimated orientation angle in degrees
    """
    # Compute angles
    angles = np.arctan2(vectors[..., 1], vectors[..., 0]) * 180/np.pi # shape (n, k)
    angles =  (angles+90)%180 - 90 # Take modulo 180 and bring to [-90, 90)
        
    # Create histogram of angles
    hist, bins = np.histogram(angles, bins=360, range=(-90, 90))
    
    # Apply circular smoothing
    hist_smooth = uniform_filter1d(hist.astype(float), size=90, mode='wrap') # even size puts convolution center at i-0.5
    
    # Find peak in smoothed histogram
    idx_peak = np.argmax(hist_smooth)
    orientation = bins[idx_peak]

    if show:
        # Plot histogram and smoothed
        plt.stairs(hist, bins, fill=True, label='Histogram', alpha=0.7)
        plt.plot((bins[1:]+bins[:-1])/2, hist_smooth, linewidth=1, label='Smoothed')
        
        # Mark detected orientation
        plt.axvline(x=orientation, color='tab:green', linestyle='--', 
                    label=f'Detected Orientation: {orientation:.1f}°')
        
        plt.title('Histogram of Neighbor Angles')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    return orientation


# %%
theta = estimate_orientation(vects_ngbr, show=True)


# %%
def estimate_spacing(vectors: List[List[Tuple[int, float, float]]], 
                     orientation: float, angle_tolerance=30, show=True) -> Tuple[float, float]:
    """
    Estimate within-line and between-line spacing as described in the paper.
    
    Args:
        neighbors_info: List of neighbor information
        orientation: Estimated text orientation
        
    Returns:
        Tuple of (within_line_spacing, between_line_spacing)
    """
    # Admissible angles
    angles = np.arctan2(vectors[..., 1], vectors[..., 0]) * 180/np.pi # shape (n, k)
    conds_wit = (np.abs((angles-orientation+90)%180-90) < angle_tolerance)
    conds_bet = (np.abs((angles-orientation)%180-90) < angle_tolerance)

    # Distances of neighbors with admissible angles
    dists = np.linalg.norm(vectors, axis=-1)  # shape (n, k)
    dists_wit = dists[conds_wit]
    dists_bet = dists[conds_bet]

    # Within-line histogram
    hist, bins = np.histogram(dists_wit, bins=np.arange(0, dists_wit.max()+0.5, 0.5))
    hist_smooth = uniform_filter1d(hist.astype(float), size=20, mode='mirror')

    # Find peak in smoothed histogram
    idx_peak = np.argmax(hist_smooth)
    spacing_wit = bins[idx_peak]

    if show:
        # Plot histogram and smoothed
        plt.stairs(hist, bins, fill=True, label='Histogram', alpha=0.7)
        plt.plot((bins[1:]+bins[:-1])/2, hist_smooth, linewidth=1, label='Smoothed')
        
        # Mark detected orientation
        plt.axvline(x=spacing_wit, color='tab:green', linestyle='--', 
                    label=f'Within-line spacing: {spacing_wit:.1f}px')
        
        plt.title('Histogram of within-line distances')
        plt.xlabel('Distance (pixels)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # Between-line histogram
    hist, bins = np.histogram(dists_bet, bins=np.arange(0, dists_bet.max()+0.5, 0.5))
    hist_smooth = uniform_filter1d(hist.astype(float), size=20, mode='mirror')

    # Find peak in smoothed histogram
    idx_peak = np.argmax(hist_smooth)
    spacing_bet = bins[idx_peak]

    if show:
        # Plot histogram and smoothed
        plt.stairs(hist, bins, fill=True, label='Histogram', alpha=0.7)
        plt.plot((bins[1:]+bins[:-1])/2, hist_smooth, linewidth=1, label='Smoothed')
        
        # Mark detected orientation
        plt.axvline(x=spacing_bet, color='tab:green', linestyle='--', 
                    label=f'Between-line spacing: {spacing_bet:.1f}px')
        
        plt.title('Histogram of between-line distances')
        plt.xlabel('Distance (pixels)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return spacing_wit, spacing_bet


# %%
spacing_wit, spacing_bet = estimate_spacing(vects_ngbr, theta, show=True)


# %%
def graph_connected_components(graph: list[list[int]]) -> list[set[int]]:
    """Find connected components in an undirected graph (V, E).
    
    Args:
        graph: Represented as [[j for j in V if (i, j) in E] for i in V],
            where V is seen as range(len(V)).
        
    Returns:
        List of connected components.
    """
    # Symmetrize the graph
    graph_rev = [[] for _ in range(len(graph))]
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            graph_rev[j].append(i)
    graph_sym = [set(ngbrs).union(ngbrs_rev)
                 for ngbrs, ngbrs_rev in zip(graph, graph_rev)]

    # Find connected components
    visited = [False for _ in range(len(graph_sym))]
    components = []
    for i in range(len(graph_sym)):
        if visited[i]:
            continue
        # Depth-first search starting from i
        line = set()
        queue = [i]
        while queue:
            j = queue.pop()
            if visited[j]:
                continue
            visited[j] = True
            line.add(j)
            queue.extend(graph_sym[j])
        components.append(sorted(list(line)))
    return components


def find_text_lines(components: List[Component], 
                    idxs_neighbors: List[List[Tuple[int, float, float]]], 
                    vectors: float,
                    spacing_wit,
                    spacing_bet,
                    orientation,
                    angle_tolerance=30) -> List[List[int]]:
    """
    Group components into text lines as described in the paper.
    
    Args:
        components: List of components
        neighbors_info: List of neighbor information
        orientation: Estimated text orientation
            
    Returns:
        List of text lines, where each line is a list of component indices
    """
    # Angles and distances
    angles = np.arctan2(vectors[..., 1], vectors[..., 0]) * 180/np.pi # shape (n, k)
    conds_angle = (np.abs((angles-orientation+90)%180-90) < angle_tolerance)
    
    dists = np.linalg.norm(vectors, axis=-1) # shape (n, k)
    conds_dist = (dists < min(3*spacing_wit, np.sqrt(3)*spacing_bet))
    
    # Build graph (V = text component indices, E = within-line connections)
    graph = [idxs_neighbors[i, (conds_angle[i] & conds_dist[i])].tolist() 
             for i in range(len(components))]

    # Find text lines as connected components of the graph, of size at least 2
    text_lines_idxs = [idxs for idxs in graph_connected_components(graph)
                       if len(idxs) >= 2]

    # Save lines as segments [P, Q]
    text_lines_PQ = []

    for line in text_lines_idxs:

        # Perform linear regression:
        # find a, b, c with a**2+b**2=1 minimizing sum((a*x_i + b*y_i + c)**2)
        pts = np.array([components[idx].centroid for idx in line]) # shape (m, 2)
        # Center points; then optimal line has c = 0
        bary = np.mean(pts, axis=0)
        pts -= bary
        # Optimal (a, b) is smallest eigenvector of covariance matrix
        S = pts.T @ pts # shape (2, 2)
        _, eigvecs = np.linalg.eigh(S)
        a, b = eigvecs[:, 0]
        if b < 0:
            a, b = -a, -b
        c = -([a, b] @ bary).item()

        # Position of points along to the line, left to right
        pos = pts @ [b, -a]
        # P (resp. Q) is the left-most (res. right-most) point
        PQ = bary+pts[[np.argmin(pos), np.argmax(pos)]] # shape (l, 2) with l=2
        # Project onto the line
        PQ -= (np.dot(PQ, [a, b]) + c) * np.array([a, b])
        text_lines_PQ.append(PQ)

    return text_lines_idxs, np.array(text_lines_PQ) # shape (m', 2, 2)


# %%
text_lines, PQs = find_text_lines(comps, idxs_ngbr, vects_ngbr,
                      spacing_bet, spacing_wit, theta)


# %%
def visualize_lines(img_binary: np.ndarray,
                    components: List[Component], 
                    PQs):
    """
    Visualize and save k-nearest neighbors connections between components
    
    Args:
        image: Original grayscale image
        components: List of components
        neighbors_info: List of neighbor information
        output_dir: Output directory path
        filename: Base filename for the output
    """
    # Create RGB visualization image
    img_vis = np.stack([(~img_binary).astype(np.uint8)*255] * 3, axis=-1)
    
    # Draw lines
    for i, (P, Q) in enumerate(PQs):
        color = tuple(int(np.rint(c*255))
                          for c in plt.cm.tab10((i/10)%1)[:3])
        cv2.line(img_vis,
                 tuple(map(int, np.rint(P))),
                 tuple(map(int, np.rint(Q))),
                 color, 1, cv2.LINE_AA)
    
    # Draw component centroids on top
    for comp in components:
        x, y = tuple(map(int, np.rint(comp.centroid)))
        img_vis[y, x] = (255, 0, 0)
    
    return img_vis


# %%
img_lines = visualize_lines(img_binary, comps, PQs)
plt.imshow(img_lines)
vpv(img_lines)


# %%
def compute_line_distances(PQ1: np.ndarray,
                           PQ2: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    # Matrix (P1, P2, Q1, Q2), shape (..., 2, 4)
    PQPQ = np.concatenate([PQ1.swapaxes(-2, -1), PQ2.swapaxes(-2, -1)],
                          axis=-1)
    
    # Rotation matrix of angle -theta1
    delta1 = PQPQ[..., 1] - PQPQ[..., 0] # Q1-P1, shape (..., 2)
    n1 = delta1 / np.linalg.norm(delta1, axis=-1, keepdims=True) # [cos, sin], shape (..., 2)
    Mrot = np.stack([n1, n1[..., ::-1]*[-1, 1]], axis=-2) # [[cos, sin], [-sin, cos]], shape (..., 2, 2)
    # Rotate points
    PPQQ = Mrot @ PQPQ

    # cos(theta2-theta1)
    delta2 = PQPQ[..., 3] - PQPQ[..., 2] # Q2-P2 after rotation, shape (..., 2)
    cos_t12 = delta2[..., 0] / np.linalg.norm(delta2, axis=-1) # shape (...)

    # Order A, B, C, D
    order = np.argsort(PQPQ[..., 0, :], axis=-1) # shape (..., 4)

    # Parallel distance
    inds = np.indices(PQPQ.shape[:-2])
    da = (PQPQ[*inds, 0, order[*inds, 2]] - PPQQ[*inds, 0, order[*inds, 1]]) / np.abs(cos_t12) # (xC-xB)/cos(theta2-theta1), shape (...)
    # Overlap condition
    xLR = np.sort(PQPQ[..., 0, [0, 1]], axis=-1) # (xP1, xQ1) in order, shape (..., 2)
    cond1 = xLR[..., 1] == PQPQ[*inds, 0, order[*inds, 1]] # xR1 == xB
    cond2 = xLR[..., 0] == PQPQ[*inds, 0, order[*inds, 2]] # xL1 == xC
    da *= np.where(cond1 | cond2, -1, 1)

    # Perpendicular distance
    xM = (PQPQ[*inds, 0, order[*inds, 1]] + PQPQ[*inds, 0, order[*inds, 2]]) / 2 # (xB+xC)/2, shape (...)
    alpha = (xM - PQPQ[..., 0, 2]) / (PQPQ[..., 0, 3] - PQPQ[..., 0, 2]) # (xM-xP2)/(xQ2-xP2), shape (...)
    # alpha = 1-alpha
    de = (1-alpha)*PQPQ[..., 1, 2] + alpha*PQPQ[..., 1, 3] - PPQQ[..., 1, 0] # (1-alpha)*yP2 + alpha*yQ2 - yP1, shape (...)
    de = np.abs(de)
    
    # Angular difference
    theta1 = np.arctan

    return da, de, cos_t12


def find_text_blocks(PQs, da_max, de_max, angle_max=30):
    # Index pairs of lines
    idxs_line1, idxs_line2 = np.triu_indices(len(PQs), k=-1)

    # Distances: parallel, perpendicular, angular
    da, de, cos = compute_line_distances(PQs[idxs_line1], PQs[idxs_line2])

    # Build graph (V = lines, E = line proximity)
    cond = ( (da >= da_max)
           & (de <= de_max)
           & (np.abs(cos) > np.cos(angle_max*np.pi/180)) )
    graph = [[] for _ in range(len(PQs))]
    for i, j in zip(idxs_line1[cond], idxs_line2[cond]):
        graph[i].append(j.item())
        graph[j].append(i.item())

    # Find text blocks as connected components of the graph
    text_blocks_idxs = graph_connected_components(graph)

    return text_blocks_idxs, graph, da, de, cos


# %%
text_blocks, graph, da, de, cos = find_text_blocks(PQs,
                               da_max=1.5*spacing_wit,
                               de_max=1.3*spacing_bet,
                               angle_max=30)


# %%
def visualize_text_blocks(img_binary: np.ndarray,
                          components: List[Component], 
                          blocks,
                          lines,
                          PQs):
    """
    Visualize and save text blocks
    
    Args:
        image: Original grayscale image
        components: List of components
        PQs: List of line segments
        text_blocks: List of text blocks, each block is a list of line indices
    """
    # Create RGB visualization image
    img_vis = np.stack([(~img_binary).astype(np.uint8)*255] * 3, axis=-1)
    
    # Draw text blocks
    for i, block in enumerate(blocks):
        color = tuple(int(np.rint(c*255))
                          for c in plt.cm.tab10((i/10)%1)[:3])

        # Draw block bounding box
        xywh = np.stack([components[j].bbox for i in block for j in lines[i]],
                        axis=0)
        ltrb = xywh @ (np.diag([1, 1, 1, 1]) + np.diag([1, 1], k=2))
        l, t = ltrb[:, 0].min(), ltrb[:, 1].min()
        r, b = ltrb[:, 2].max(), ltrb[:, 3].max()
        cv2.rectangle(img_vis, (l, t), (r, b), color, 1)

        # Draw lines
        for j in block:
            P, Q = PQs[j]
            cv2.line(img_vis,
                     tuple(map(int, np.rint(P))),
                     tuple(map(int, np.rint(Q))),
                     color, 1, cv2.LINE_AA)
        
        # Draw component boxes
        for l, t, r, b in ltrb:
            cv2.rectangle(img_vis, (l, t), (r, b), color, 1)
    
    # Draw component centroids on top
    for comp in components:
        x, y = tuple(map(int, np.rint(comp.centroid)))
        img_vis[y, x] = (255, 0, 0)
    
    return img_vis


# %%
img_blocks = visualize_text_blocks(img_binary, comps, text_blocks, text_lines, PQs)
plt.imshow(img_blocks)
vpv(img_blocks)
