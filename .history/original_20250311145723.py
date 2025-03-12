import argparse
import cv2
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
import os
import matplotlib
matplotlib.use('Agg')

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
        
        # Invert if needed (assuming text is black)
        if np.mean(binary) > 127:
            binary = 255 - binary
            
        # Convert to binary format (0 and 1)
        binary = (binary > 0).astype(np.uint8)
        
        return binary
        
    def find_connected_components(self, binary: np.ndarray) -> List[Component]:
        """
        Find connected components in binary image
        
        Args:
            binary: Binary image
            
        Returns:
            List of Component objects
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        components = []
        # Skip background component (index 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            components.append(Component(
                bbox=(x, y, w, h),
                centroid=(centroids[i][0], centroids[i][1]),
                area=area
            ))
        
        return components

    def find_nearest_neighbors(self, components: List[Component]) -> List[List[Tuple[int, float, float]]]:
        """
        Find k nearest neighbors for each component using KD-Tree
        
        Args:
            components: List of components
            
        Returns:
            List of lists containing (neighbor_idx, distance, angle) tuples for each component
        """
        # Extract centroids
        points = np.array([c.centroid for c in components])
        
        # Build KD-tree for efficient nearest neighbor search
        tree = KDTree(points)
        
        # Find k nearest neighbors (first one is the point itself)
        distances, indices = tree.query(points, k=self.k+1)
        
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

    def estimate_spacing(self, neighbors_info: List[List[Tuple[int, float, float]]], 
                         orientation: float) -> Tuple[float, float]:
        """
        Estimate within-line and between-line spacing
        
        Args:
            neighbors_info: List of neighbor information
            orientation: Estimated text orientation
            
        Returns:
            Tuple of (within_line_spacing, between_line_spacing)
        """
        # Collect distances for within-line neighbors
        within_line_distances = []
        between_line_distances = []
        
        for component_neighbors in neighbors_info:
            for _, dist, angle in component_neighbors:
                # Check if angle is within threshold of orientation (within-line)
                angle_diff = min((angle - orientation) % 180, (orientation - angle) % 180)
                if angle_diff < self.angle_threshold:
                    within_line_distances.append(dist)
                # Check if angle is within threshold of perpendicular to orientation (between-line)
                elif angle_diff > 90 - self.angle_threshold and angle_diff < 90 + self.angle_threshold:
                    between_line_distances.append(dist)
        
        # Histogram analysis for more accurate spacing estimation
        if within_line_distances:
            within_hist, within_bins = np.histogram(within_line_distances, bins=50)
            within_line_spacing = within_bins[np.argmax(within_hist)]
        else:
            within_line_spacing = 0
            
        if between_line_distances:
            between_hist, between_bins = np.histogram(between_line_distances, bins=50)
            between_line_spacing = between_bins[np.argmax(between_hist)]
        else:
            between_line_spacing = 0
            
        return within_line_spacing, between_line_spacing

    def find_text_lines(self, components: List[Component], 
                        neighbors_info: List[List[Tuple[int, float, float]]], 
                        orientation: float) -> List[List[int]]:
        """
        Group components into text lines
        
        Args:
            components: List of components
            neighbors_info: List of neighbor information
            orientation: Estimated text orientation
                
        Returns:
            List of text lines, where each line is a list of component indices
        """
        # Create graph of connected components
        n = len(components)
        graph = {i: [] for i in range(n)}
        
        # Connect components that are within the angle threshold of orientation
        for i, component_neighbors in enumerate(neighbors_info):
            for neighbor_idx, _, angle in component_neighbors:
                # Check if angle is within threshold of orientation
                angle_diff = min((angle - orientation) % 180, (orientation - angle) % 180)
                if angle_diff < self.angle_threshold:
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
        
        for i in range(n):
            if i not in visited:
                current_line = []
                dfs(i, current_line)
                # Sort components in line by x-coordinate for left-to-right reading
                current_line.sort(key=lambda idx: components[idx].centroid[0])
                text_lines.append(current_line)
        
        # Sort text lines by y-coordinate (top to bottom)
        text_lines.sort(key=lambda line: min(components[idx].centroid[1] for idx in line))
        
        return text_lines

    def find_blocks(self, components: List[Component], text_lines: List[List[int]], 
                   within_line_spacing: float, between_line_spacing: float) -> List[List[List[int]]]:
        """
        Group text lines into blocks based on the docstrum criteria
        
        Args:
            components: List of components
            text_lines: List of text lines
            within_line_spacing: Estimated within-line spacing
            between_line_spacing: Estimated between-line spacing
            
        Returns:
            List of blocks, where each block is a list of text lines
        """
        if not text_lines:
            return []
        
        # Calculate perpendicular distance threshold (typically 1.3 times between-line spacing)
        perpendicular_threshold = 1.3 * between_line_spacing
        
        # Calculate parallel distance threshold (typically 1.5 times within-line spacing)
        parallel_threshold = 1.5 * within_line_spacing
        
        # Calculate minimum overlap ratio
        min_overlap_ratio = 0.1
        
        def get_line_bounds(line):
            """Get bounding box of a text line"""
            x1 = min(components[idx].bbox[0] for idx in line)
            y1 = min(components[idx].bbox[1] for idx in line)
            x2 = max(components[idx].bbox[0] + components[idx].bbox[2] for idx in line)
            y2 = max(components[idx].bbox[1] + components[idx].bbox[3] for idx in line)
            return (x1, y1, x2, y2)
        
        def perpendicular_distance(bounds1, bounds2):
            """Calculate perpendicular distance between lines"""
            _, y1, _, y2 = bounds1
            _, y3, _, y4 = bounds2
            return min(abs(y2 - y3), abs(y1 - y4))
        
        def parallel_distance(bounds1, bounds2):
            """Calculate parallel distance between lines"""
            x1, _, x2, _ = bounds1
            x3, _, x4, _ = bounds2
            if x2 < x3:  # bounds1 is to the left of bounds2
                return x3 - x2
            elif x4 < x1:  # bounds2 is to the left of bounds1
                return x1 - x4
            else:  # overlap
                return 0
        
        def horizontal_overlap_ratio(bounds1, bounds2):
            """Calculate horizontal overlap ratio"""
            x1, _, x2, _ = bounds1
            x3, _, x4, _ = bounds2
            overlap = min(x2, x4) - max(x1, x3)
            if overlap <= 0:
                return 0
            width1 = x2 - x1
            width2 = x4 - x3
            return overlap / min(width1, width2)
        
        # Calculate bounds for each text line
        line_bounds = [get_line_bounds(line) for line in text_lines]
        
        # Initialize blocks
        blocks = []
        current_block = [text_lines[0]]
        current_bounds = line_bounds[0]
        
        # Group lines into blocks
        for i in range(1, len(text_lines)):
            line = text_lines[i]
            bounds = line_bounds[i]
            
            # Check criteria for grouping
            perp_dist = perpendicular_distance(current_bounds, bounds)
            para_dist = parallel_distance(current_bounds, bounds)
            overlap = horizontal_overlap_ratio(current_bounds, bounds)
            
            # Apply docstrum criteria:
            # 1. Perpendicular proximity
            # 2. Either overlap or parallel proximity
            if (perp_dist <= perpendicular_threshold and
                (overlap >= min_overlap_ratio or para_dist <= parallel_threshold)):
                current_block.append(line)
                # Update current block bounds
                x1 = min(current_bounds[0], bounds[0])
                y1 = min(current_bounds[1], bounds[1])
                x2 = max(current_bounds[2], bounds[2])
                y2 = max(current_bounds[3], bounds[3])
                current_bounds = (x1, y1, x2, y2)
            else:
                blocks.append(current_block)
                current_block = [line]
                current_bounds = bounds
        
        blocks.append(current_block)
        return blocks

    def process(self, image: np.ndarray):
        """
        Process image with the docstrum algorithm
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple containing components, text lines, orientation, blocks, and spacings
        """
        # Preprocess image
        binary = self.preprocess(image)
        
        # Find connected components
        components = self.find_connected_components(binary)
        
        # Find k-nearest neighbors
        neighbors_info = self.find_nearest_neighbors(components)
        
        # Estimate orientation
        orientation = self.estimate_orientation(neighbors_info)
        
        # Estimate spacing
        within_line_spacing, between_line_spacing = self.estimate_spacing(neighbors_info, orientation)
        
        # Find text lines
        text_lines = self.find_text_lines(components, neighbors_info, orientation)
        
        # Find blocks
        blocks = self.find_blocks(components, text_lines, within_line_spacing, between_line_spacing)
        
        return components, text_lines, orientation, blocks, (within_line_spacing, between_line_spacing)
    
    def visualize_results(self, image: np.ndarray, components: List[Component], 
                        text_lines: List[List[int]], blocks: List[List[List[int]]],
                        orientation: float, spacings: Tuple[float, float],
                        output_dir: str, filename: str):
        """
        Visualize results of docstrum analysis
        
        Args:
            image: Original grayscale image
            components: List of components
            text_lines: List of text lines
            blocks: List of blocks
            orientation: Estimated orientation angle
            spacings: Tuple of (within_line_spacing, between_line_spacing)
            output_dir: Output directory path
            filename: Base filename for the output
        """
        within_line_spacing, between_line_spacing = spacings
        
        # Create RGB visualization image
        vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        
        # Generate distinct colors for blocks
        colors = plt.cm.tab10(np.linspace(0, 1, len(blocks)))
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
            max_x = max(components[idx].bbox[0] + components[idx].bbox[2] for idx in block_components)
            max_y = max(components[idx].bbox[1] + components[idx].bbox[3] for idx in block_components)
            
            # Draw block rectangle
            padding = 3
            cv2.rectangle(vis_image, 
                        (min_x - padding, min_y - padding), 
                        (max_x + padding, max_y + padding), 
                        color, 2)
        
        # Save the visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{filename}_blocks.png')
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Create and save a plot with detailed information
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Docstrum Analysis Results\nOrientation: {orientation:.1f}°, '
                 f'Within-line spacing: {within_line_spacing:.1f}, '
                 f'Between-line spacing: {between_line_spacing:.1f}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}_analysis.png'), dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run Docstrum page layout analysis on images.')
    parser.add_argument('input_path', type=str, 
                       help='Path to input image or directory containing images')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save output visualizations (default: output)')
    parser.add_argument('--k_nearest', type=int, default=5, 
                       help='Number of nearest neighbors (default: 5)')
    parser.add_argument('--angle_threshold', type=float, default=30, 
                       help='Angle threshold in degrees (default: 30)')
    
    args = parser.parse_args()
    
    # Initialize docstrum
    docstrum = Docstrum(k_nearest=args.k_nearest, angle_threshold=args.angle_threshold)
    
    # Process single image or directory
    if os.path.isfile(args.input_path):
        # Single image processing
        image = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not load image {args.input_path}")
            return
        
        filename = os.path.splitext(os.path.basename(args.input_path))[0]
        components, text_lines, orientation, blocks, spacings = docstrum.process(image)
        docstrum.visualize_results(image, components, text_lines, blocks, orientation, spacings, 
                                 args.output_dir, filename)
        
        print(f"Processed {filename}:")
        print(f"- Found {len(components)} components")
        print(f"- Grouped into {len(text_lines)} text lines")
        print(f"- Detected {len(blocks)} text blocks")
        print(f"- Estimated orientation: {orientation:.1f}°")
        print(f"- Within-line spacing: {spacings[0]:.1f}")
        print(f"- Between-line spacing: {spacings[1]:.1f}")
        print(f"- Output saved to {args.output_dir}")
        
    elif os.path.isdir(args.input_path):
        # Directory processing
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        processed = 0
        
        for filename in os.listdir(args.input_path):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_path = os.path.join(args.input_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Error: Could not load image {image_path}")
                    continue
                
                base_filename = os.path.splitext(filename)[0]
                components, text_lines, orientation, blocks, spacings = docstrum.process(image)
                docstrum.visualize_results(image, components, text_lines, blocks, orientation, spacings, 
                                         args.output_dir, base_filename)
                
                print(f"Processed {filename}:")
                print(f"- Found {len(components)} components")
                print(f"- Grouped into {len(text_lines)} text lines")
                print(f"- Detected {len(blocks)} text blocks")
                print(f"- Estimated orientation: {orientation:.1f}°")
                print(f"- Output saved to {args.output_dir}")
                
                processed += 1
        
        print(f"\nProcessed {processed} images. Output saved to {args.output_dir}")
        
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")

if __name__ == '__main__':
    main()