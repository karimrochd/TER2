import argparse
import cv2
import sys
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass
import os
import matplotlib
matplotlib.use('Agg')


def remove_small_components(binary_image, small_component_threshold = 7):
    """
    Remove connected components smaller than a given size.

    Args:
        binary_image (numpy.ndarray): Binary image (1 for foreground, 0 for background).
        small_component_threshold (int): Minimum size of connected components to retain.

    Returns:
        numpy.ndarray: Binary image with small components removed.
    """
    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an output image initialized to 0
    output_image = np.zeros_like(binary_image, dtype=np.uint8)
    
    # Iterate over each component
    for i in range(1, num_labels):  # Skip the background label (0)
        if stats[i, cv2.CC_STAT_AREA] >= small_component_threshold:
            # Retain components larger than or equal to k
            output_image[labels == i] = 1
    
    return output_image


@dataclass
class Component:
    """Class to store connected component information"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[float, float]
    area: int

class Docstrum:
    def __init__(self, k_nearest: int = 5, angle_threshold: float = 30):
        """
        Initialize docstrum processor
        
        Args:
            k_nearest: Number of nearest neighbors to find (default 5)
            angle_threshold: Angle threshold in degrees for within-line connections
        """
        self.k = k_nearest
        self.angle_threshold = angle_threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image - noise reduction and binarization
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary image
        """
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary = (binary == 0).astype(np.uint8)

        binary = remove_small_components(binary, small_component_threshold=7)
        
        return binary

    def find_connected_components(self, binary: np.ndarray) -> List[Component]:
        """
        Find connected components in binary image and filter them based on size
        
        Args:
            binary: Binary image
            
        Returns:
            List of Component objects
        """
        # Invert binary image if needed (assuming text is black)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
            
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        if num_labels < 2:  # 1 is background
            raise ValueError("No components found in the image")
        
        # Calculate median area to use for filtering
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        median_area = np.median(areas)
        
        components = []
        # Skip background component (index 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out components that are too small or too large
            if area < median_area * 0.05:
                continue
                
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            components.append(Component(
                bbox=(x, y, w, h),
                centroid=(centroids[i][0], centroids[i][1]),
                area=area
            ))
        
        if not components:
            raise ValueError("No valid components found after filtering")
            
        print(f"Found {len(components)} valid components")
        return components

    def find_nearest_neighbors(self, components: List[Component]) -> List[List[Tuple[int, float, float]]]:
        """
        Find k nearest neighbors for each component
        
        Args:
            components: List of components
            
        Returns:
            List of lists containing (neighbor_idx, distance, angle) tuples for each component
        """
        if len(components) < self.k + 1:
            raise ValueError(f"Not enough components ({len(components)}) for k={self.k} nearest neighbors")
            
        # Extract centroids
        points = np.array([c.centroid for c in components])
        
        # Adjust k if necessary
        k = min(self.k + 1, len(components))
        print(f"Finding {k-1} nearest neighbors for each component")
        
        # Build KD-tree for efficient nearest neighbor search
        tree = KDTree(points)
        
        # Find k nearest neighbors (first one is the point itself)
        distances, indices = tree.query(points, k=k)
        
        neighbors_info = []
        for i, (component_neighbors, neighbor_distances) in enumerate(zip(indices, distances)):
            # Skip the first neighbor (point itself)
            neighbors = []
            for j, (neighbor_idx, dist) in enumerate(zip(component_neighbors[1:], neighbor_distances[1:]), 1):
                # Calculate angle between components
                dx = points[neighbor_idx][0] - points[i][0]
                dy = points[neighbor_idx][1] - points[i][1]
                angle = np.degrees(np.arctan2(dy, dx)) % 180
                
                neighbors.append((neighbor_idx, dist, angle))
            
            neighbors_info.append(neighbors)
            
        return neighbors_info


    def estimate_orientation(self, neighbors_info: List[List[Tuple[int, float, float]]]) -> float:
        """
        Estimate document orientation from neighbor angles
        
        Args:
            neighbors_info: List of neighbor information
            
        Returns:
            Estimated orientation angle in degrees
        """
        # Collect all angles
        angles = []
        for component_neighbors in neighbors_info:
            angles.extend([n[2] for n in component_neighbors])
            
        # Create histogram of angles
        hist, bins = np.histogram(angles, bins=180, range=(0, 180))
        
        # Apply smoothing to histogram
        hist = np.convolve(hist, np.ones(5)/5, mode='same')
        
        # Find peak
        orientation = bins[np.argmax(hist)]
        
        return orientation


    def find_text_lines(self, components: List[Component], 
                    neighbors_info: List[List[Tuple[int, float, float]]], 
                    orientation: float,
                    spacing_factor: float = 1.2) -> List[List[int]]:
        """
        Group components into text lines, considering local intercharacter spacing
        
        Args:
            components: List of components
            neighbors_info: List of neighbor information
            orientation: Estimated text orientation
            spacing_factor: Factor to multiply local intercharacter space for max allowed gap
                
        Returns:
            List of text lines, where each line is a list of component indices
        """
        def calculate_local_spacing(component_idx: int, potential_neighbors: List[int]) -> float:
            """Calculate median intercharacter spacing for a component and its aligned neighbors"""
            if not potential_neighbors:
                return float('inf')
                
            # Get horizontal gaps between component and its aligned neighbors
            gaps = []
            comp = components[component_idx]
            
            for n_idx in potential_neighbors:
                neighbor = components[n_idx]
                
                # Determine which component is leftmost
                if comp.centroid[0] < neighbor.centroid[0]:
                    left, right = comp, neighbor
                    left_idx, right_idx = component_idx, n_idx
                else:
                    left, right = neighbor, comp
                    left_idx, right_idx = n_idx, component_idx
                    
                # Calculate gap between components
                gap = (right.bbox[0] - (left.bbox[0] + left.bbox[2]))
                if gap > 0:  # Only consider positive gaps
                    gaps.append(gap)
                    
            return np.median(gaps) if gaps else float('inf')
        
        # Create graph of connected components
        graph = {i: [] for i in range(len(components))}
        
        # First pass: Find potentially aligned components
        aligned_components = {i: [] for i in range(len(components))}
        for i, component_neighbors in enumerate(neighbors_info):
            for neighbor_idx, dist, angle in component_neighbors:
                # Check if angle is within threshold of orientation
                angle_diff = min((angle - orientation) % 180, (orientation - angle) % 180)
                if angle_diff < self.angle_threshold:
                    aligned_components[i].append(neighbor_idx)
                    aligned_components[neighbor_idx].append(i)
        
        # Second pass: Apply local spacing constraints
        for i in range(len(components)):
            # Calculate local spacing for current component
            local_spacing = calculate_local_spacing(i, aligned_components[i])
            
            # Add edges only if components are within local spacing constraint
            for neighbor_idx in aligned_components[i]:
                comp = components[i]
                neighbor = components[neighbor_idx]
                
                # Calculate horizontal distance between components
                if comp.centroid[0] < neighbor.centroid[0]:
                    left, right = comp, neighbor
                else:
                    left, right = neighbor, comp
                    
                distance = right.bbox[0] - (left.bbox[0] + left.bbox[2])
                
                # Only add edge if distance is within local spacing constraint
                if distance <= local_spacing * spacing_factor:
                    graph[i].append(neighbor_idx)
                    graph[neighbor_idx].append(i)
        
        # Find connected components in graph (text lines)
        text_lines = []
        visited = set()
        
        def dfs(node: int, current_line: List[int]):
            visited.add(node)
            current_line.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_line)
        
        for i in range(len(components)):
            if i not in visited:
                current_line = []
                dfs(i, current_line)
                # Sort components in line by x-coordinate for left-to-right reading
                current_line.sort(key=lambda idx: components[idx].centroid[0])
                text_lines.append(current_line)
        
        # Sort text lines by y-coordinate (top to bottom)
        text_lines.sort(key=lambda line: min(components[idx].centroid[1] for idx in line))
        
        return text_lines
    

    def merge_overlapping_blocks(self, components: List[Component], blocks: List[List[List[int]]], 
                                horizontal_distance_threshold: float = 50,
                                vertical_distance_threshold: float = 50,
                                just_lines: bool = True) -> List[List[List[int]]]:
        """
        Merge blocks based on configuration and containment:
        - Merge blocks that are contained within other blocks
        - If just_lines is True: only merge blocks in the same line that are horizontally close
        - If just_lines is False: also merge blocks that are vertically close
        
        Args:
            components: List of components
            blocks: List of blocks (each block contains text lines)
            horizontal_distance_threshold: Maximum allowed horizontal distance between blocks to merge
            vertical_distance_threshold: Maximum allowed vertical distance between blocks to merge
            just_lines: If True, only merge blocks in the same line
                
        Returns:
            List of merged blocks
        """
        def get_block_bounds(block):
            """Get the bounding box of a block"""
            block_components = [comp_idx for line in block for comp_idx in line]
            if not block_components:
                return None
                
            min_x = min(components[idx].bbox[0] for idx in block_components)
            min_y = min(components[idx].bbox[1] for idx in block_components)
            max_x = max(components[idx].bbox[0] + components[idx].bbox[2] for idx in block_components)
            max_y = max(components[idx].bbox[1] + components[idx].bbox[3] for idx in block_components)
            
            return (min_x, min_y, max_x, max_y)

        def blocks_are_in_same_line(bounds1, bounds2, vertical_tolerance=0.5):
            """Check if two blocks are roughly in the same line"""
            _, y1, _, y2 = bounds1
            _, y3, _, y4 = bounds2
            
            height1 = y2 - y1
            height2 = y4 - y3
            min_height = min(height1, height2)
            
            center1 = (y1 + y2) / 2
            center2 = (y3 + y4) / 2
            return abs(center1 - center2) < min_height * vertical_tolerance

        def horizontal_distance(bounds1, bounds2):
            """Calculate horizontal distance between two blocks"""
            x1, _, x2, _ = bounds1
            x3, _, x4, _ = bounds2
            
            if x2 >= x3 and x1 <= x4:
                return 0
            return min(abs(x2 - x3), abs(x1 - x4))

        def vertical_distance(bounds1, bounds2):
            """Calculate vertical distance between two blocks"""
            _, y1, _, y2 = bounds1
            _, y3, _, y4 = bounds2
            
            if y2 >= y3 and y1 <= y4:
                return 0
            return min(abs(y2 - y3), abs(y1 - y4))

        def horizontal_overlap_exists(bounds1, bounds2, tolerance=0.3):
            """Check if blocks have some horizontal overlap"""
            x1, _, x2, _ = bounds1
            x3, _, x4, _ = bounds2
            
            overlap = min(x2, x4) - max(x1, x3)
            if overlap <= 0:
                return False
                
            width1 = x2 - x1
            width2 = x4 - x3
            min_width = min(width1, width2)
            
            return overlap >= min_width * tolerance

        while True:
            merged = False
            block_bounds = [get_block_bounds(block) for block in blocks]
            
            # Check each pair of blocks
            for i in range(len(blocks)):
                if i >= len(blocks):  # Check if block was removed
                    continue
                    
                for j in range(i + 1, len(blocks)):
                    if j >= len(blocks):  # Check if block was removed
                        continue
                        
                    bounds1 = block_bounds[i]
                    bounds2 = block_bounds[j]
                    
                    if bounds1 is None or bounds2 is None:
                        continue
                    
                    should_merge = False
                    merge_order = 0  # 0: normal merge, -1: i into j, 1: j into i
                    
                    # First check containment
                    containment = check_block_containment(bounds1, bounds2)
                    if containment != 0:
                        should_merge = True
                        merge_order = containment
                    else:
                        # Check horizontal merging (same line)
                        if blocks_are_in_same_line(bounds1, bounds2) and \
                        horizontal_distance(bounds1, bounds2) <= horizontal_distance_threshold:
                            should_merge = True
                        
                        # Check vertical merging if not just_lines
                        elif not just_lines and \
                            vertical_distance(bounds1, bounds2) <= vertical_distance_threshold and \
                            horizontal_overlap_exists(bounds1, bounds2):
                            should_merge = True
                    
                    if should_merge:
                        if merge_order == -1:  # bounds1 is contained within bounds2
                            blocks[j].extend(blocks[i])
                            blocks.pop(i)
                            block_bounds.pop(i)
                        else:  # bounds2 is contained within bounds1 or normal merge
                            blocks[i].extend(blocks[j])
                            blocks.pop(j)
                            block_bounds.pop(j)
                        merged = True
                        break
                
                if merged:
                    break
                    
            if not merged:
                break
        
        return blocks

    def visualize_results(self, image: np.ndarray, components: List[Component], 
                        text_lines: List[List[int]], orientation: float):
        """
        Visualize results with individual line blocks
        """
        # Create RGB visualization image
        vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        
        # Create blocks (one per line)
        blocks = self.find_blocks(components, text_lines)
        print(f"Found {len(blocks)} line blocks")
        
        # Generate distinct colors for blocks
        colors = plt.cm.Set3(np.linspace(0, 1, len(blocks)))
        colors = (colors[:, :3] * 255).astype(int)
        
        # Draw blocks (lines)
        for block_idx, block in enumerate(blocks):
            # Each block contains exactly one line
            line = block[0]  # Get the single line from the block
            
            if not line:
                continue
            
            # Find line boundaries
            min_x = min(components[idx].bbox[0] for idx in line)
            min_y = min(components[idx].bbox[1] for idx in line)
            max_x = max(components[idx].bbox[0] + components[idx].bbox[2] 
                    for idx in line)
            max_y = max(components[idx].bbox[1] + components[idx].bbox[3] 
                    for idx in line)
            
            # Draw line rectangle with padding
            color = colors[block_idx % len(colors)].tolist()
            padding = 3
            cv2.rectangle(vis_image, 
                        (min_x - padding, min_y - padding), 
                        (max_x + padding, max_y + padding), 
                        color, 2)
            

        
        plt.figure(figsize=(15, 10))
        plt.imshow(vis_image)
        plt.title(f'Detected Text Lines (Orientation: {orientation:.1f}°)')
        plt.axis('off')
        plt.show()


    def find_blocks(self, components: List[Component], text_lines: List[List[int]], 
                max_vertical_gap: float = 1.5, horizontal_overlap_threshold: float = 0.2) -> List[List[List[int]]]:
        """
        Improved block detection with stricter merging criteria
        
        Args:
            components: List of components
            text_lines: List of text lines
            max_vertical_gap: Maximum vertical gap multiplier (smaller = stricter)
            horizontal_overlap_threshold: Required horizontal overlap ratio (larger = stricter)
        """
        if not text_lines:
            return []
            
        def get_line_bounds(line):
            x1 = min(components[idx].bbox[0] for idx in line)
            y1 = min(components[idx].bbox[1] for idx in line)
            x2 = max(components[idx].bbox[0] + components[idx].bbox[2] for idx in line)
            y2 = max(components[idx].bbox[1] + components[idx].bbox[3] for idx in line)
            return (x1, y1, x2, y2)

        def get_line_height(line):
            heights = [components[idx].bbox[3] for idx in line]
            return np.median(heights)
        
        def horizontal_overlap_ratio(bounds1, bounds2):
            x1, _, x2, _ = bounds1
            x3, _, x4, _ = bounds2
            overlap = min(x2, x4) - max(x1, x3)
            if overlap <= 0:
                return 0
            width1 = x2 - x1
            width2 = x4 - x3
            return overlap / min(width1, width2)
        
        # Calculate typical line spacing
        line_spacings = []
        line_bounds = [get_line_bounds(line) for line in text_lines]
        line_heights = [get_line_height(line) for line in text_lines]
        
        for i in range(len(text_lines) - 1):
            _, _, _, y2 = line_bounds[i]
            _, y3, _, _ = line_bounds[i + 1]
            line_spacings.append(y3 - y2)
        
        if not line_spacings:
            return [text_lines]
            
        median_spacing = np.median(line_spacings)
        median_height = np.median(line_heights)
        
        # Group lines into initial blocks
        blocks = []
        current_block = [text_lines[0]]
        current_bounds = line_bounds[0]
        
        for i in range(1, len(text_lines)):
            current_line = text_lines[i]
            current_line_bounds = line_bounds[i]
            
            # Check vertical gap
            _, _, _, prev_bottom = current_bounds
            _, curr_top, _, _ = current_line_bounds
            vertical_gap = curr_top - prev_bottom
            
            # Check horizontal overlap with previous line
            overlap = horizontal_overlap_ratio(current_bounds, current_line_bounds)
            
            # Stricter merging conditions:
            # 1. Must have significant horizontal overlap
            # 2. Vertical gap must be reasonable
            # 3. Consider line height in spacing calculation
            if (overlap > horizontal_overlap_threshold and 
                vertical_gap <= max(median_spacing * max_vertical_gap, median_height * 1.5)):
                current_block.append(current_line)
                # Update bounds
                x1 = min(current_bounds[0], current_line_bounds[0])
                y1 = min(current_bounds[1], current_line_bounds[1])
                x2 = max(current_bounds[2], current_line_bounds[2])
                y2 = max(current_bounds[3], current_line_bounds[3])
                current_bounds = (x1, y1, x2, y2)
            else:
                blocks.append(current_block)
                current_block = [current_line]
                current_bounds = current_line_bounds
        
        blocks.append(current_block)
        return blocks



def check_block_containment(bounds1, bounds2, tolerance=0.9):
    """
    Check if one block is contained within another block
    
    Args:
        bounds1: (min_x1, min_y1, max_x1, max_y1) of first block
        bounds2: (min_x2, min_y2, max_x2, max_y2) of second block
        tolerance: How much overlap is required to consider it containment (0-1)
        
    Returns:
        -1 if bounds1 is contained within bounds2
        1 if bounds2 is contained within bounds1
        0 if no containment
    """
    x1, y1, x2, y2 = bounds1
    x3, y3, x4, y4 = bounds2
    
    # Calculate areas
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    # Calculate intersection
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Check if one block is contained within the other
    if intersection_area >= area1 * tolerance:
        return -1  # bounds1 is contained within bounds2
    elif intersection_area >= area2 * tolerance:
        return 1  # bounds2 is contained within bounds1
    
    return 0


def visualize_preprocessing(image: np.ndarray, binary: np.ndarray):
    """
    Visualize the preprocessing step showing original and binary images side by side
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Show original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Show binary image
    ax2.imshow(binary, cmap='gray')
    ax2.set_title('Preprocessed Binary Image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_components(image: np.ndarray, components: List[Component]):
    """
    Visualize detected connected components with bounding boxes and centroids
    """
    # Create RGB visualization image
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    
    # Generate distinct colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(components)))
    colors = (colors[:, :3] * 255).astype(int)
    
    # Draw components
    for idx, comp in enumerate(components):
        color = colors[idx % len(colors)].tolist()
        x, y, w, h = comp.bbox
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 1)
        
        # Draw centroid
        cx, cy = map(int, comp.centroid)
        cv2.circle(vis_image, (cx, cy), 2, color, -1)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(vis_image)
    plt.title(f'Detected Components (Total: {len(components)})')
    plt.axis('off')
    plt.show()

def visualize_neighbors(image: np.ndarray, components: List[Component], 
                    neighbors_info: List[List[Tuple[int, float, float]]]):
    """
    Visualize k-nearest neighbors connections between components
    """
    # Create RGB visualization image
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    
    # Draw connections between components
    for i, component_neighbors in enumerate(neighbors_info):
        x1, y1 = map(int, components[i].centroid)
        
        for neighbor_idx, dist, angle in component_neighbors:
            x2, y2 = map(int, components[neighbor_idx].centroid)
            
            # Color based on angle (cyclic color map)
            color = plt.cm.hsv(angle / 180)[:3]
            color = tuple(int(c * 255) for c in color)
            
            # Draw line connecting components
            cv2.line(vis_image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
    # Draw component centroids on top
    for comp in components:
        cx, cy = map(int, comp.centroid)
        cv2.circle(vis_image, (cx, cy), 2, (255, 0, 0), -1)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(vis_image)
    plt.title('K-Nearest Neighbors Connections')
    plt.axis('off')
    plt.show()

def visualize_orientation_histogram(neighbors_info: List[List[Tuple[int, float, float]]], 
                                orientation: float):
    """
    Visualize histogram of angles and detected orientation
    """
    # Collect all angles
    angles = []
    for component_neighbors in neighbors_info:
        angles.extend([n[2] for n in component_neighbors])
        
    # Create histogram
    plt.figure(figsize=(12, 6))
    hist, bins, _ = plt.hist(angles, bins=180, range=(0, 180), 
                            color='skyblue', alpha=0.7)
    
    # Apply smoothing for visualization
    smoothed = np.convolve(hist, np.ones(5)/5, mode='same')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, smoothed, 'r-', linewidth=2, label='Smoothed')
    
    # Mark detected orientation
    plt.axvline(x=orientation, color='green', linestyle='--', 
                label=f'Detected Orientation: {orientation:.1f}°')
    
    plt.title('Histogram of Neighbor Angles')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_text_lines(image: np.ndarray, components: List[Component], 
                        text_lines: List[List[int]]):
    """
    Visualize detected text lines with different colors
    """
    # Create RGB visualization image
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    
    # Generate distinct colors for lines
    colors = plt.cm.rainbow(np.linspace(0, 1, len(text_lines)))
    colors = (colors[:, :3] * 255).astype(int)
    
    # Draw text lines
    for line_idx, line in enumerate(text_lines):
        color = colors[line_idx % len(colors)].tolist()
        
        # Draw bounding boxes for components in line
        for comp_idx in line:
            x, y, w, h = components[comp_idx].bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Connect components in line
        if len(line) > 1:
            for i in range(len(line) - 1):
                x1, y1 = components[line[i]].centroid
                x2, y2 = components[line[i + 1]].centroid
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.line(vis_image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(vis_image)
    plt.title(f'Detected Text Lines (Total: {len(text_lines)})')
    plt.axis('off')
    plt.show()

def visualize_initial_blocks(image: np.ndarray, components: List[Component], 
                        blocks: List[List[List[int]]]):
    """
    Visualize initial text blocks before merging
    """
    # Create RGB visualization image
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    
    # Generate distinct colors for blocks
    colors = plt.cm.Set3(np.linspace(0, 1, len(blocks)))
    colors = (colors[:, :3] * 255).astype(int)
    
    # Draw blocks
    for block_idx, block in enumerate(blocks):
        color = colors[block_idx % len(colors)].tolist()
        
        # Get all components in block
        block_components = [comp_idx for line in block for comp_idx in line]
        
        if not block_components:
            continue
            
        # Find block boundaries
        min_x = min(components[idx].bbox[0] for idx in block_components)
        min_y = min(components[idx].bbox[1] for idx in block_components)
        max_x = max(components[idx].bbox[0] + components[idx].bbox[2] 
                for idx in block_components)
        max_y = max(components[idx].bbox[1] + components[idx].bbox[3] 
                for idx in block_components)
        
        # Draw block rectangle
        padding = 3
        cv2.rectangle(vis_image, 
                    (min_x - padding, min_y - padding), 
                    (max_x + padding, max_y + padding), 
                    color, 2)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(vis_image)
    plt.title(f'Initial Text Blocks (Total: {len(blocks)})')
    plt.axis('off')
    plt.show()

def calculate_vertical_threshold(text_lines: List[List[int]], components: List[Component]) -> float:
    """
    Calculate optimal vertical threshold based on line spacing histogram
    
    Args:
        text_lines: List of text lines (each line is a list of component indices)
        components: List of components
        
    Returns:
        float: Calculated vertical threshold
    """
    if len(text_lines) < 2:
        return 7.0  # Default value if insufficient lines
        
    # Calculate vertical distances between consecutive lines
    distances = []
    for i in range(len(text_lines) - 1):
        current_line = text_lines[i]
        next_line = text_lines[i + 1]
        
        # Get bottom of current line and top of next line
        current_bottom = max(components[idx].bbox[1] + components[idx].bbox[3] for idx in current_line)
        next_top = min(components[idx].bbox[1] for idx in next_line)
        
        distances.append(next_top - current_bottom)
    
    if not distances:
        return 7.0
        
    # Create histogram of distances
    hist, bins = np.histogram(distances, bins='auto')
    
    # Find the most common distance range
    peak_idx = np.argmax(hist)
    most_common_distance = abs(bins[peak_idx] + bins[peak_idx + 1]) / 2
    
    # Apply a safety factor  to account for slight variations
    return most_common_distance * 1.2


# def calculate_horizontal_threshold(text_lines: List[List[int]], components: List[Component]) -> float:
#     """
#     Calculate optimal horizontal threshold based on within-line spacing histogram
    
#     Args:
#         text_lines: List of text lines (each line is a list of component indices)
#         components: List of components
        
#     Returns:
#         float: Calculated horizontal threshold
#     """
#     if not text_lines:
#         return 12.0  # Default value if no lines
        
#     # Calculate horizontal distances between consecutive components in each line
#     distances = []
#     for line in text_lines:
#         if len(line) < 2:
#             continue
            
#         # Sort components in line by x-coordinate
#         sorted_components = sorted(line, key=lambda idx: components[idx].bbox[0])
        
#         for i in range(len(sorted_components) - 1):
#             current_idx = sorted_components[i]
#             next_idx = sorted_components[i + 1]
            
#             # Get right edge of current component and left edge of next component
#             current_right = components[current_idx].bbox[0] + components[current_idx].bbox[2]
#             next_left = components[next_idx].bbox[0]
            
#             distance = abs(next_left - current_right)
#             if distance > 0:  # Only include positive distances
#                 distances.append(distance)
    
#     if not distances:
#         return 12.0
        
#     # Create histogram of distances
#     hist, bins = np.histogram(distances, bins='auto')
    
#     # Find the most common distance range
#     peak_idx = np.argmax(hist)
#     most_common_distance = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    
#     # Use a larger safety factor  for horizontal distances to account for varying word spacing
#     return most_common_distance * 3.0




def process_and_save_visualization(image: np.ndarray, output_dir: str, filename: str, 
                                 docstrum: Docstrum, spacing_factor: float, 
                                 horizontal_distance_threshold: float,
                                 vertical_distance_threshold: float, just_lines: bool):
    """
    Process an image and save final visualization
    
    Args:
        image: Input grayscale image
        output_dir: Directory to save visualization
        filename: Base filename for saving visualization
        docstrum: Initialized Docstrum object
        spacing_factor: Factor to multiply local intercharacter space for max allowed gap
        corner_threshold: Maximum distance between corners to consider them as sharing a border
        horizontal_distance_threshold: Maximum horizontal distance between blocks to merge them
        vertical_distance_threshold: Maximum vertical distance between blocks to merge them
        just_lines: If True, only merge blocks in the same line
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process image with spacing_factor
    binary = docstrum.preprocess(image)
    components = docstrum.find_connected_components(binary)
    neighbors_info = docstrum.find_nearest_neighbors(components)
    orientation = docstrum.estimate_orientation(neighbors_info)
    text_lines = docstrum.find_text_lines(components, neighbors_info, orientation, spacing_factor=spacing_factor)
    initial_blocks = docstrum.find_blocks(components, text_lines)


    if vertical_distance_threshold == -1:
        vertical_distance_threshold = calculate_vertical_threshold(text_lines, components)
        print(f"Automatically calculated vertical threshold: {vertical_distance_threshold:.2f}")
    
    # if horizontal_distance_threshold == -1:
    #     horizontal_distance_threshold = calculate_horizontal_threshold(text_lines, components)
    #     print(f"Automatically calculated horizontal threshold: {horizontal_distance_threshold:.2f}")
    

    merged_blocks = docstrum.merge_overlapping_blocks(
        components, initial_blocks, 
        horizontal_distance_threshold=horizontal_distance_threshold,
        vertical_distance_threshold=vertical_distance_threshold,
        just_lines=just_lines
    )
    
    # Create final visualization
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    colors = plt.cm.Set3(np.linspace(0, 1, len(merged_blocks)))
    colors = (colors[:, :3] * 255).astype(int)
    
    for block_idx, block in enumerate(merged_blocks):
        block_components = [comp_idx for line in block for comp_idx in line]
        if not block_components:
            continue
        
        min_x = min(components[idx].bbox[0] for idx in block_components)
        min_y = min(components[idx].bbox[1] for idx in block_components)
        max_x = max(components[idx].bbox[0] + components[idx].bbox[2] for idx in block_components)
        max_y = max(components[idx].bbox[1] + components[idx].bbox[3] for idx in block_components)
        
        color = colors[block_idx % len(colors)].tolist()
        padding = 3
        cv2.rectangle(vis_image, 
                     (min_x - padding, min_y - padding), 
                     (max_x + padding, max_y + padding), 
                     color, 2)
    
    # Save the visualization
    output_path = os.path.join(output_dir, f'{filename}_blocks.png')
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return components, text_lines, orientation, merged_blocks
def main():
    parser = argparse.ArgumentParser(description='Run Docstrum page layout analysis on images.')
    parser.add_argument('input_path', type=str, 
                       help='Path to input image or directory containing images')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save output visualizations (default: output)')
    parser.add_argument('--k_nearest', type=int, default=5, 
                       help='Number of nearest neighbors (default: 5)')
    parser.add_argument('--angle_threshold', type=float, default=5, 
                       help='Angle threshold in degrees (default: 5)')
    # the default spacing factor (1.2) works well for most cases if it doesn't work well for your images, you can try to adjust horizontal_distance_threshold, it was left as a parameter for flexibility for some edge cases
    parser.add_argument('--spacing_factor', type=float, default=1.2,  
                       help='Factor to multiply local intercharacter space for max allowed gap (default: 1.2)')
    parser.add_argument('--horizontal_distance_threshold', type=float, default=12,
                       help='Maximum horizontal distance between blocks to merge them (default: 12)')
    parser.add_argument('--vertical_distance_threshold', type=float, default= -1,
                       help='Maximum vertical distance between blocks to merge them (default: -1), -1 to use the calculate the threshold automaticaly, not needed for lines')
    parser.add_argument('--just_lines', action='store_true', default=False,
                       help='If True, only merge blocks in the same line (default: False)')
    

    
    args = parser.parse_args()
    
    # Initialize docstrum
    docstrum = Docstrum(k_nearest=args.k_nearest, angle_threshold=args.angle_threshold)
    
    # Process single image or directory
    if os.path.isfile(args.input_path):
        # Single image processing
        image = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not load image {args.input_path}")
            sys.exit(1)
        
        filename = os.path.splitext(os.path.basename(args.input_path))[0]
        try:
            components, text_lines, orientation, blocks = process_and_save_visualization(
                image, args.output_dir, filename, docstrum, 
                args.spacing_factor, 
                args.horizontal_distance_threshold,
                args.vertical_distance_threshold,
                args.just_lines
            )
            print(f"Processed {filename}:")
            print(f"- Found {len(components)} components")
            print(f"- Grouped into {len(text_lines)} text lines")
            print(f"- Detected {len(blocks)} text blocks")
            print(f"- Estimated orientation: {orientation:.1f} degrees")
            print(f"- Merge mode: {'line-only' if args.just_lines else 'lines and vertical'}")
            print(f"- Saved visualization to {args.output_dir}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            
    elif os.path.isdir(args.input_path):
        # Directory processing
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        processed = 0
        errors = 0
        
        for filename in os.listdir(args.input_path):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_path = os.path.join(args.input_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Error: Could not load image {image_path}")
                    errors += 1
                    continue
                
                base_filename = os.path.splitext(filename)[0]
                try:
                    components, text_lines, orientation, blocks = process_and_save_visualization(
                        image, args.output_dir, base_filename, docstrum,
                        args.spacing_factor,
                        args.horizontal_distance_threshold,
                        args.vertical_distance_threshold,
                        args.just_lines
                    )
                    print(f"Processed {filename}:")
                    print(f"- Found {len(components)} components")
                    print(f"- Grouped into {len(text_lines)} text lines")
                    print(f"- Detected {len(blocks)} text blocks")
                    print(f"- Estimated orientation: {orientation:.1f} degrees")
                    print(f"- Merge mode: {'line-only' if args.just_lines else 'lines and vertical'}")
                    processed += 1
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    errors += 1
        
        print(f"\nProcessing complete:")
        print(f"- Successfully processed: {processed} images")
        print(f"- Errors: {errors} images")
        print(f"- Output saved to: {args.output_dir}")
        
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")
        sys.exit(1)

if __name__ == '__main__':
    main()




