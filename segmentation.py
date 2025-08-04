"""
Core Docstrum algorithm implementation for document layout analysis.

This module implements the Docstrum algorithm for page segmentation,
including text line detection and block formation.
"""

import cv2
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import convolve1d
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from preprocess import binarize_image, remove_small_components


@dataclass
class Component:
    """
    Connected component information.
    
    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) - top-left and bottom-right coordinates
        centroid: Center point of the component as (x, y)
        area: Area of the bounding box in pixels
        
    Note:
        Use Component.from_cv2_stats() to create from OpenCV's connectedComponentsWithStats output,
        which automatically converts from (x, y, w, h) to (x1, y1, x2, y2) format.
    """
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]
    area: int
    
    @classmethod
    def from_cv2_stats(cls, stats: np.ndarray, centroid: np.ndarray, 
                      index: int) -> 'Component':
        """
        Create Component from OpenCV connectedComponentsWithStats output.
        
        Args:
            stats: Stats array from cv2.connectedComponentsWithStats
            centroid: Centroid array from cv2.connectedComponentsWithStats  
            index: Component index
            
        Returns:
            Component instance
        """
        x = stats[index, cv2.CC_STAT_LEFT]
        y = stats[index, cv2.CC_STAT_TOP]
        w = stats[index, cv2.CC_STAT_WIDTH]
        h = stats[index, cv2.CC_STAT_HEIGHT]
        
        return cls(
            bbox=(x, y, x + w, y + h),
            centroid=(centroid[index][0], centroid[index][1]),
            area=w * h
        )


class Docstrum:
    """
    Docstrum algorithm for document layout analysis.
    
    The Docstrum algorithm uses k-nearest neighbor analysis of connected
    components to detect text lines and blocks in document images.
    """
    
    def __init__(self, k_nearest: int = 5, angle_threshold: float = 5.0):
        """
        Initialize Docstrum processor.
        
        Args:
            k_nearest: Number of nearest neighbors to analyze
            angle_threshold: Angle threshold in degrees for within-line connections
        """
        self.k = k_nearest
        self.angle_threshold = angle_threshold
    
    def preprocess(self, image: np.ndarray, 
                  small_component_threshold: float = 0.05,
                  binarization_threshold: int = -1,
                  kfill_threshold: int = 5,
                  filter_type: int = 2,
                  kfill_iterations: int = 10) -> np.ndarray:
        """
        Preprocess the image with binarization and noise reduction.
        
        Args:
            image: Input grayscale image
            small_component_threshold: Minimum size threshold for components
            binarization_threshold: Threshold for binarization (-1 for Otsu)
            kfill_threshold: Window size for kFill filter
            filter_type: Filtering type (0: kFill, 1: size, 2: both)
            kfill_iterations: Maximum kFill iterations
            
        Returns:
            Preprocessed binary image
        """
        # Binarize
        binary = binarize_image(image, binarization_threshold)
        
        # Remove small components
        binary = remove_small_components(
            binary,
            small_component_threshold=small_component_threshold,
            kfill_threshold=kfill_threshold,
            filter_type=filter_type,
            kfill_iterations=kfill_iterations
        )
        
        return binary
    
    def find_connected_components(self, binary: np.ndarray,
                                big_component_threshold: int = -1) -> List[Component]:
        """
        Find and filter connected components in binary image.
        
        Components are created with bounding boxes in (x1, y1, x2, y2) format.
        
        Args:
            binary: Binary image (text as 1, background as 0)
            big_component_threshold: Maximum size threshold (-1 to disable)
            
        Returns:
            List of Component objects with bbox in (x1, y1, x2, y2) format
            
        Raises:
            ValueError: If no valid components found
        """
        # Ensure text is white (1) on black (0) background
        if np.mean(binary) > 0.5:
            binary = cv2.bitwise_not(binary)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        if num_labels < 2:  # Only background
            raise ValueError("No components found in the image")
        
        # Calculate size distribution
        bbox_areas = [stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT] 
                     for i in range(1, num_labels)]
        
        # Find most common size
        hist, bin_edges = np.histogram(bbox_areas, bins='auto')
        peak_bin_index = np.argmax(hist)
        peak_area = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
        
        components = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter out too large components
            if big_component_threshold != -1 and area > peak_area * big_component_threshold:
                continue
            
            components.append(Component.from_cv2_stats(stats, centroids, i))
        
        if not components:
            raise ValueError("No valid components found after filtering")
        
        print(f"Found {len(components)} valid components")
        return components
    
    def find_nearest_neighbors(self, components: List[Component]) -> List[List[Tuple[int, float, float]]]:
        """
        Find k nearest neighbors for each component using KD-tree.
        
        Args:
            components: List of components
            
        Returns:
            List of neighbor information (index, distance, angle) for each component
            
        Raises:
            ValueError: If insufficient components for k-NN analysis
        """
        if len(components) < self.k + 1:
            raise ValueError(f"Not enough components ({len(components)}) for k={self.k} nearest neighbors")
        
        # Extract centroids
        points = np.array([c.centroid for c in components])
        
        # Build KD-tree
        k = min(self.k + 1, len(components))
        tree = KDTree(points)
        
        # Find k nearest neighbors
        distances, indices = tree.query(points, k=k)
        
        neighbors_info = []
        for i, (component_neighbors, neighbor_distances) in enumerate(zip(indices, distances)):
            neighbors = []
            # Skip first neighbor (point itself)
            for neighbor_idx, dist in zip(component_neighbors[1:], neighbor_distances[1:]):
                # Calculate angle
                dx = points[neighbor_idx][0] - points[i][0]
                dy = points[neighbor_idx][1] - points[i][1]
                angle = np.degrees(np.arctan2(dy, dx)) % 180
                
                neighbors.append((neighbor_idx, dist, angle))
            
            neighbors_info.append(neighbors)
        
        return neighbors_info
    
    def estimate_orientation(self, smoothing_window: int,
                           neighbors_info: List[List[Tuple[int, float, float]]]) -> float:
        """
        Estimate document orientation from neighbor angle distribution.
        
        Args:
            smoothing_window: Size of smoothing window (must be odd)
            neighbors_info: Neighbor information from find_nearest_neighbors
            
        Returns:
            Estimated orientation angle in degrees
        """
        # Collect all angles
        angles = []
        for component_neighbors in neighbors_info:
            angles.extend([n[2] for n in component_neighbors])
        
        # Create angle histogram
        hist, bins = np.histogram(angles, bins=360, range=(0, 180))
        
        # Ensure odd window size
        if smoothing_window % 2 == 0:
            smoothing_window += 1
        
        # Smooth histogram
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = convolve1d(hist, kernel, mode='wrap')
        
        # Find peak
        orientation = bins[np.argmax(smoothed)]
        
        return orientation
    
    def find_text_lines(self, components: List[Component],
                       neighbors_info: List[List[Tuple[int, float, float]]],
                       orientation: float,
                       spacing_factor: float = 1.2) -> List[List[int]]:
        """
        Group components into text lines based on local spacing analysis.
        
        Uses the Docstrum algorithm's approach of analyzing local intercharacter
        spacing to determine which components belong to the same text line.
        
        Args:
            components: List of components with bbox in (x1, y1, x2, y2) format
            neighbors_info: Neighbor information from find_nearest_neighbors
            orientation: Estimated text orientation in degrees
            spacing_factor: Factor for maximum allowed gap (default: 1.2)
            
        Returns:
            List of text lines, where each line is a list of component indices
            sorted left-to-right, and lines are sorted top-to-bottom
        """
        def calculate_local_spacing(component_idx: int, potential_neighbors: List[int]) -> float:
            """Calculate median intercharacter spacing for aligned neighbors."""
            if not potential_neighbors:
                return float('inf')
            
            gaps = []
            comp = components[component_idx]
            
            for n_idx in potential_neighbors:
                neighbor = components[n_idx]
                
                # Determine left-right order
                if comp.centroid[0] < neighbor.centroid[0]:
                    left, right = comp, neighbor
                else:
                    left, right = neighbor, comp
                
                # Calculate horizontal gap between bounding boxes
                gap = right.bbox[0] - left.bbox[2]
                if gap > 0:
                    gaps.append(gap)
            
            return np.median(gaps) if gaps else float('inf')
        
        # Build alignment graph
        graph = {i: [] for i in range(len(components))}
        aligned_components = {i: [] for i in range(len(components))}
        
        # Find aligned components
        for i, component_neighbors in enumerate(neighbors_info):
            for neighbor_idx, dist, angle in component_neighbors:
                angle_diff = min((angle - orientation) % 180, 
                               (orientation - angle) % 180)
                if angle_diff < self.angle_threshold:
                    aligned_components[i].append(neighbor_idx)
                    aligned_components[neighbor_idx].append(i)
        
        # Apply local spacing constraints
        for i in range(len(components)):
            local_spacing = calculate_local_spacing(i, aligned_components[i])
            
            for neighbor_idx in aligned_components[i]:
                comp = components[i]
                neighbor = components[neighbor_idx]
                
                # Calculate horizontal distance between bounding boxes
                if comp.centroid[0] < neighbor.centroid[0]:
                    left, right = comp, neighbor
                else:
                    left, right = neighbor, comp
                
                distance = right.bbox[0] - left.bbox[2]
                
                # Add edge if within threshold
                if distance <= local_spacing * spacing_factor:
                    graph[i].append(neighbor_idx)
                    graph[neighbor_idx].append(i)
        
        # Find connected components (text lines) using DFS
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
                # Sort by x-coordinate
                current_line.sort(key=lambda idx: components[idx].centroid[0])
                text_lines.append(current_line)
        
        # Sort lines by y-coordinate
        text_lines.sort(key=lambda line: min(components[idx].centroid[1] for idx in line))
        
        return text_lines
    
    def find_blocks(self, components: List[Component], text_lines: List[List[int]],
                   max_vertical_gap: float = 1.5, 
                   horizontal_overlap_threshold: float = 0.2) -> List[List[List[int]]]:
        """
        Group text lines into blocks based on proximity and alignment.
        
        Args:
            components: List of components
            text_lines: List of text lines
            max_vertical_gap: Maximum vertical gap multiplier
            horizontal_overlap_threshold: Required horizontal overlap ratio
            
        Returns:
            List of blocks (each block contains multiple text lines)
        """
        if not text_lines:
            return []
        
        def get_line_bounds(line):
            """Get bounding box of a line as (x1, y1, x2, y2)."""
            x1 = min(components[idx].bbox[0] for idx in line)
            y1 = min(components[idx].bbox[1] for idx in line)
            x2 = max(components[idx].bbox[2] for idx in line)
            y2 = max(components[idx].bbox[3] for idx in line)
            return (x1, y1, x2, y2)
        
        def get_line_height(line):
            """Get median height of components in a line."""
            heights = [components[idx].bbox[3] - components[idx].bbox[1] for idx in line]
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
        
        # Calculate line properties
        line_bounds = [get_line_bounds(line) for line in text_lines]
        line_heights = [get_line_height(line) for line in text_lines]
        
        # Calculate typical spacing
        line_spacings = []
        for i in range(len(text_lines) - 1):
            _, _, _, y2 = line_bounds[i]
            _, y3, _, _ = line_bounds[i + 1]
            line_spacings.append(y3 - y2)
        
        median_spacing = np.median(line_spacings) if line_spacings else 0
        median_height = np.median(line_heights)
        
        # Group lines into blocks
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
            
            # Check horizontal overlap
            overlap = horizontal_overlap_ratio(current_bounds, current_line_bounds)
            
            # Decide if line belongs to current block
            if (overlap > horizontal_overlap_threshold and 
                vertical_gap <= max(median_spacing * max_vertical_gap, 
                                  median_height * 1.5)):
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
    
    def merge_overlapping_blocks(self, components: List[Component],
                               blocks: List[List[List[int]]],
                               horizontal_distance_threshold: float = 50,
                               vertical_distance_threshold: float = 50,
                               just_lines: bool = True,
                               block_overlap_threshold: float = 0.9) -> List[List[List[int]]]:
        """
        Merge blocks based on containment and proximity criteria.
        
        This method merges blocks in two phases:
        1. Containment and horizontal merging (always applied)
        2. Vertical merging (only if just_lines=False)
        
        All bounding boxes use (x1, y1, x2, y2) format.
        
        Args:
            components: List of components
            blocks: List of blocks to merge
            horizontal_distance_threshold: Max horizontal distance for merging
            vertical_distance_threshold: Max vertical distance for merging
            just_lines: If True, only merge horizontally aligned blocks
            block_overlap_threshold: Overlap ratio for containment detection
            
        Returns:
            List of merged blocks
        """
        def get_block_bounds(block):
            """Get bounding box of a block as (x1, y1, x2, y2)."""
            block_components = [comp_idx for line in block for comp_idx in line]
            if not block_components:
                return None
            
            min_x = min(components[idx].bbox[0] for idx in block_components)
            min_y = min(components[idx].bbox[1] for idx in block_components)
            max_x = max(components[idx].bbox[2] for idx in block_components)
            max_y = max(components[idx].bbox[3] for idx in block_components)
            
            return (min_x, min_y, max_x, max_y)
        
        def blocks_are_in_same_line(bounds1, bounds2, vertical_tolerance=1.0):
            """Check if two blocks are roughly in the same line."""
            _, y1, _, y2 = bounds1
            _, y3, _, y4 = bounds2
            
            height1 = y2 - y1
            height2 = y4 - y3
            min_height = min(height1, height2)
            
            center1 = (y1 + y2) / 2
            center2 = (y3 + y4) / 2
            return abs(center1 - center2) < min_height * vertical_tolerance
        
        def horizontal_distance(bounds1, bounds2):
            """Calculate horizontal distance between blocks."""
            x1, _, x2, _ = bounds1
            x3, _, x4, _ = bounds2
            
            if x2 >= x3 and x1 <= x4:
                return 0
            return min(abs(x2 - x3), abs(x1 - x4))
        
        def vertical_distance(bounds1, bounds2):
            """Calculate vertical distance between blocks."""
            _, y1, _, y2 = bounds1
            _, y3, _, y4 = bounds2
            
            if y2 >= y3 and y1 <= y4:
                return 0
            return min(abs(y2 - y3), abs(y1 - y4))
        
        def horizontal_overlap_exists(bounds1, bounds2, tolerance=0.3):
            """Check if blocks have horizontal overlap."""
            x1, _, x2, _ = bounds1
            x3, _, x4, _ = bounds2
            
            overlap = min(x2, x4) - max(x1, x3)
            if overlap <= 0:
                return False
            
            width1 = x2 - x1
            width2 = x4 - x3
            min_width = min(width1, width2)
            
            return overlap >= min_width * tolerance
        
        # Phase 1: Handle containment and horizontal merging
        while True:
            merged = False
            block_bounds = [get_block_bounds(block) for block in blocks]
            
            for i in range(len(blocks)):
                if i >= len(blocks):
                    continue
                
                for j in range(i + 1, len(blocks)):
                    if j >= len(blocks):
                        continue
                    
                    bounds1 = block_bounds[i]
                    bounds2 = block_bounds[j]
                    
                    if bounds1 is None or bounds2 is None:
                        continue
                    
                    should_merge = False
                    merge_order = 0
                    
                    # Check containment
                    containment = check_block_containment(bounds1, bounds2, 
                                                        block_overlap_threshold)
                    if containment != 0:
                        should_merge = True
                        merge_order = containment
                    # Check horizontal merging
                    elif (blocks_are_in_same_line(bounds1, bounds2) and 
                          horizontal_distance(bounds1, bounds2) <= horizontal_distance_threshold):
                        should_merge = True
                    
                    if should_merge:
                        if merge_order == -1:  # bounds1 contained in bounds2
                            blocks[j].extend(blocks[i])
                            blocks.pop(i)
                            block_bounds.pop(i)
                        else:  # Normal merge or bounds2 contained in bounds1
                            blocks[i].extend(blocks[j])
                            blocks.pop(j)
                            block_bounds.pop(j)
                        merged = True
                        break
                
                if merged:
                    break
            
            if not merged:
                break
        
        # Phase 2: Vertical merging (only if not just_lines)
        if not just_lines:
            while True:
                merged = False
                block_bounds = [get_block_bounds(block) for block in blocks]
                
                for i in range(len(blocks)):
                    if i >= len(blocks):
                        continue
                    
                    for j in range(i + 1, len(blocks)):
                        if j >= len(blocks):
                            continue
                        
                        bounds1 = block_bounds[i]
                        bounds2 = block_bounds[j]
                        
                        if bounds1 is None or bounds2 is None:
                            continue
                        
                        # Check vertical merging criteria
                        if (vertical_distance(bounds1, bounds2) <= vertical_distance_threshold and 
                            horizontal_overlap_exists(bounds1, bounds2)):
                            # Merge the blocks
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


def check_block_containment(bounds1: Tuple[float, float, float, float],
                          bounds2: Tuple[float, float, float, float],
                          tolerance: float = 0.9) -> int:
    """
    Check if one block is contained within another.
    
    Args:
        bounds1: Bounding box (x1, y1, x2, y2) of first block
        bounds2: Bounding box (x1, y1, x2, y2) of second block
        tolerance: Required overlap ratio for containment (0-1)
        
    Returns:
        -1 if bounds1 is contained in bounds2
         1 if bounds2 is contained in bounds1
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
    
    # Check containment
    if intersection_area >= area1 * tolerance:
        return -1  # bounds1 contained in bounds2
    elif intersection_area >= area2 * tolerance:
        return 1   # bounds2 contained in bounds1
    
    return 0


def calculate_vertical_threshold(text_lines: List[List[int]], 
                               components: List[Component]) -> float:
    """
    Calculate optimal vertical threshold based on line spacing distribution.
    
    Analyzes the vertical gaps between consecutive text lines to determine
    an appropriate threshold for block merging. Uses the most common spacing
    with a safety factor.
    
    Args:
        text_lines: List of text lines (each line contains component indices)
        components: List of components with bbox in (x1, y1, x2, y2) format
        
    Returns:
        Calculated vertical threshold with 1.2x safety factor
        
    Note:
        Returns default value of 7.0 if insufficient data for calculation
    """
    if len(text_lines) < 2:
        return 7.0  # Default value
    
    # Calculate vertical distances between consecutive lines
    distances = []
    for i in range(len(text_lines) - 1):
        current_line = text_lines[i]
        next_line = text_lines[i + 1]
        
        # Get bottom of current line and top of next line
        current_bottom = max(components[idx].bbox[3] for idx in current_line)
        next_top = min(components[idx].bbox[1] for idx in next_line)
        
        distances.append(next_top - current_bottom)
    
    if not distances:
        return 7.0
    
    # Find most common distance
    hist, bins = np.histogram(distances, bins='auto')
    peak_idx = np.argmax(hist)
    most_common_distance = abs(bins[peak_idx] + bins[peak_idx + 1]) / 2
    
    # Apply safety factor
    return most_common_distance * 1.2