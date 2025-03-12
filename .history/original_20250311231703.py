import argparse
import cv2
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
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
    contour_length: int

def kfill(binary_image, k=5, max_iterations=10):
    """
    Implement the kFill filter for noise reduction in binary document images.
    This is the Will filter mentioned in the paper.
    
    Args:
        binary_image (numpy.ndarray): Binary image (1 for foreground, 0 for background).
        k (int): Window size parameter (must be odd).
        max_iterations (int): Maximum number of iterations to perform.
        
    Returns:
        numpy.ndarray: Filtered binary image.
    """
    # Ensure k is odd
    if k % 2 == 0:
        k = k + 1
    
    # Create a copy of the image
    filtered_image = binary_image.copy()
    
    iteration = 0
    changes_made = True
    
    # Continue until no changes or max iterations reached
    while changes_made and iteration < max_iterations:
        changes_made = False
        iteration += 1
        
        # Perform ON-fill and OFF-fill sub-iterations
        for fill_value in [1, 0]:  # 1 for ON-fill, 0 for OFF-fill
            height, width = filtered_image.shape
            
            # Create a copy to store changes for this sub-iteration
            temp_image = filtered_image.copy()
            
            # Process each pixel
            for y in range(k//2, height - k//2):
                for x in range(k//2, width - k//2):
                    # Extract window
                    window = filtered_image[y - k//2 : y + k//2 + 1, x - k//2 : x + k//2 + 1]
                    
                    # Define core and neighborhood
                    core = window[1:-1, 1:-1]
                    
                    # Only proceed if all core values are opposite of fill_value
                    if fill_value == 1 and np.any(core == 1):
                        continue
                    if fill_value == 0 and np.any(core == 0):
                        continue
                    
                    # Extract neighborhood (perimeter of window)
                    neighborhood = np.concatenate([
                        window[0, :],                # Top row
                        window[-1, :],               # Bottom row
                        window[1:-1, 0],             # Left column (without corners)
                        window[1:-1, -1]             # Right column (without corners)
                    ])
                    
                    # Calculate n (number of ON or OFF pixels in neighborhood)
                    if fill_value == 1:
                        n = np.sum(neighborhood == 1)  # Count ON pixels
                    else:
                        n = np.sum(neighborhood == 0)  # Count OFF pixels
                    
                    # Calculate c (number of connected groups in neighborhood)
                    # We need to analyze the neighborhood as a circular list
                    expanded_neighborhood = np.concatenate([neighborhood, neighborhood[0:1]])
                    c = 0
                    for i in range(len(neighborhood)):
                        if expanded_neighborhood[i] != expanded_neighborhood[i+1]:
                            c += 1
                    c = c // 2  # Each transition is counted twice (ON->OFF and OFF->ON)
                    
                    # Calculate r (number of corner pixels that are ON or OFF)
                    corners = [window[0, 0], window[0, -1], window[-1, 0], window[-1, -1]]
                    if fill_value == 1:
                        r = sum(1 for corner in corners if corner == 1)
                    else:
                        r = sum(1 for corner in corners if corner == 0)
                    
                    # Apply kFill condition: (c = 1) AND [(n > 3k - 4) OR (n = 3k - 4) AND r = 2]
                    if (c == 1) and ((n > 3*k - 4) or ((n == 3*k - 4) and (r == 2))):
                        # Fill the core
                        temp_image[y - k//2 + 1 : y + k//2, x - k//2 + 1 : x + k//2] = fill_value
                        changes_made = True
            
            # Update filtered_image with the results of this sub-iteration
            filtered_image = temp_image.copy()
    
    return filtered_image

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
        Preprocess the image - noise reduction and binarization as described in the paper.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary image
        """
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (assuming text is black)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
            
        # Convert to binary format (0 and 1)
        binary = (binary > 0).astype(np.uint8)
        
        # Apply the Will filter (kFill) for noise reduction as mentioned in the paper
        binary = kfill(binary, k=5, max_iterations=10)
        
        return binary
    
    def size_filtering(self, components: List[Component]) -> List[Component]:
        """
        Filter components based on size histogram as described in the paper.
        
        Args:
            components: List of components
            
        Returns:
            Filtered list of components
        """
        if not components:
            return []
            
        # Calculate size for each component (bounding box area)
        sizes = [(comp.bbox[2] * comp.bbox[3]) for comp in components]
        
        # Create histogram of sizes
        hist, bins = np.histogram(sizes, bins='auto')
        
        # Find the peak for predominant font size
        peak_idx = np.argmax(hist)
        peak_size = (bins[peak_idx] + bins[peak_idx + 1]) / 2
        
        # Filter components - keep those within a reasonable range of peak size
        # Paper suggests keeping components in the predominant size range
        filtered_components = []
        min_size = peak_size / 3  # Allow for subscripts/small characters
        max_size = peak_size * 3  # Allow for larger characters but not titles
        
        for i, comp in enumerate(components):
            if min_size <= sizes[i] <= max_size:
                filtered_components.append(comp)
        
        return filtered_components

    def find_connected_components(self, binary: np.ndarray) -> List[Component]:
        """
        Find connected components in binary image using contours as described in the paper.
        The paper mentions using "thin line code" (TLC), but we'll use OpenCV contours
        which provide the necessary features.
        
        Args:
            binary: Binary image
            
        Returns:
            List of Component objects
        """
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        components = []
        for contour in contours:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Calculate area and contour length
            area = cv2.contourArea(contour)
            contour_length = cv2.arcLength(contour, True)
            
            # Create component
            component = Component(
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area=area,
                contour_length=contour_length
            )
            
            components.append(component)
        
        # Apply size filtering as described in the paper
        filtered_components = self.size_filtering(components)
        
        return filtered_components

    def find_nearest_neighbors(self, components: List[Component]) -> List[List[Tuple[int, float, float]]]:
        """
        Find k nearest neighbors for each component, exactly as described in the paper.
        
        Args:
            components: List of components
            
        Returns:
            List of lists containing (neighbor_idx, distance, angle) tuples for each component
        """
        if len(components) < self.k + 1:
            return []
            
        # Extract centroids
        points = np.array([c.centroid for c in components])
        
        # Sort components by x position to improve efficiency as mentioned in the paper
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        reverse_mapping = {idx: i for i, idx in enumerate(sorted_indices)}
        
        # Find nearest neighbors efficiently as described in the paper
        neighbors_info = [[] for _ in range(len(components))]
        
        for i, idx in enumerate(sorted_indices):
            # Start with nearest components in x direction
            point = sorted_points[i]
            neighbors = []
            
            # Look to the left and right
            left = i - 1
            right = i + 1
            
            while len(neighbors) < self.k and (left >= 0 or right < len(sorted_points)):
                # Check left if valid
                if left >= 0:
                    left_point = sorted_points[left]
                    left_dist = np.sqrt((point[0] - left_point[0])**2 + (point[1] - left_point[1])**2)
                    left_idx = sorted_indices[left]
                    
                    # Calculate angle
                    dx = left_point[0] - point[0]
                    dy = left_point[1] - point[1]
                    angle = np.degrees(np.arctan2(dy, dx)) % 180
                    
                    neighbors.append((left_idx, left_dist, angle))
                    left -= 1
                
                # Check right if valid
                if right < len(sorted_points):
                    right_point = sorted_points[right]
                    right_dist = np.sqrt((point[0] - right_point[0])**2 + (point[1] - right_point[1])**2)
                    right_idx = sorted_indices[right]
                    
                    # Calculate angle
                    dx = right_point[0] - point[0]
                    dy = right_point[1] - point[1]
                    angle = np.degrees(np.arctan2(dy, dx)) % 180
                    
                    neighbors.append((right_idx, right_dist, angle))
                    right += 1
            
            # Sort by distance and take k nearest
            neighbors.sort(key=lambda x: x[1])
            neighbors_info[idx] = neighbors[:self.k]
        
        return neighbors_info

    def estimate_orientation(self, neighbors_info: List[List[Tuple[int, float, float]]]) -> float:
        """
        Estimate document orientation from neighbor angles as described in the paper.
        
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
        
        # Apply circular smoothing as described in the paper
        # Use rectangular smoothing window that is 25% of total range
        window_size = 45  # 25% of 180
        smoothed_hist = np.zeros_like(hist, dtype=float)
        
        for i in range(len(hist)):
            sum_val = 0
            count = 0
            for j in range(i - window_size // 2, i + window_size // 2 + 1):
                # Handle circular wrapping
                wrapped_j = j % len(hist)
                sum_val += hist[wrapped_j]
                count += 1
            smoothed_hist[i] = sum_val / count
        
        # Find peak in smoothed histogram
        peak_idx = np.argmax(smoothed_hist)
        orientation = bins[peak_idx] + (bins[peak_idx+1] - bins[peak_idx]) / 2
        
        return orientation

    def estimate_spacing(self, neighbors_info: List[List[Tuple[int, float, float]]], 
                        orientation: float) -> Tuple[float, float]:
        """
        Estimate within-line and between-line spacing as described in the paper.
        
        Args:
            neighbors_info: List of neighbor information
            orientation: Estimated text orientation
            
        Returns:
            Tuple of (within_line_spacing, between_line_spacing)
        """
        # Define angular range for within-line neighbors
        angle_tolerance = self.angle_threshold
        within_line_angles = [(orientation - angle_tolerance) % 180, 
                             (orientation + angle_tolerance) % 180]
        
        # Define angular range for between-line neighbors
        between_line_orientation = (orientation + 90) % 180
        between_line_angles = [(between_line_orientation - angle_tolerance) % 180, 
                              (between_line_orientation + angle_tolerance) % 180]
        
        # Collect distances for within-line and between-line neighbors
        within_line_distances = []
        between_line_distances = []
        
        for component_neighbors in neighbors_info:
            for _, dist, angle in component_neighbors:
                # Handle circular angle comparisons
                def is_in_range(a, range_start, range_end):
                    if range_start <= range_end:
                        return range_start <= a <= range_end
                    else:  # Range wraps around 180
                        return range_start <= a <= 180 or 0 <= a <= range_end
                
                # Check if within line
                if is_in_range(angle, within_line_angles[0], within_line_angles[1]) or \
                   is_in_range((angle + 180) % 180, within_line_angles[0], within_line_angles[1]):
                    within_line_distances.append(dist)
                
                # Check if between line
                elif is_in_range(angle, between_line_angles[0], between_line_angles[1]) or \
                     is_in_range((angle + 180) % 180, between_line_angles[0], between_line_angles[1]):
                    between_line_distances.append(dist)
        
        # Create histograms with proper resolution and smoothing as described in the paper
        def find_spacing_from_histogram(distances):
            if not distances:
                return 0
                
            # Use 2 pixels/bin resolution as mentioned in the paper
            hist, bins = np.histogram(distances, bins=np.arange(0, max(distances) + 2, 2))
            
            # Apply smoothing with a window of 5 bins
            smoothed_hist = np.convolve(hist, np.ones(5)/5, mode='same')
            
            # Find peak
            peak_idx = np.argmax(smoothed_hist)
            spacing = bins[peak_idx] + (bins[peak_idx+1] - bins[peak_idx]) / 2
            
            return spacing
        
        within_line_spacing = find_spacing_from_histogram(within_line_distances)
        between_line_spacing = find_spacing_from_histogram(between_line_distances)
        
        return within_line_spacing, between_line_spacing

    def find_text_lines(self, components: List[Component], 
                      neighbors_info: List[List[Tuple[int, float, float]]], 
                      orientation: float) -> List[List[int]]:
        """
        Group components into text lines as described in the paper.
        
        Args:
            components: List of components
            neighbors_info: List of neighbor information
            orientation: Estimated text orientation
                
        Returns:
            List of text lines, where each line is a list of component indices
        """
        # Build graph of connected components for within-line neighbors
        graph = {i: set() for i in range(len(components))}
        
        for i, component_neighbors in enumerate(neighbors_info):
            for neighbor_idx, _, angle in component_neighbors:
                # Check if angle is within threshold of orientation
                angle_diff = min((angle - orientation) % 180, (orientation - angle) % 180)
                if angle_diff < self.angle_threshold:
                    graph[i].add(neighbor_idx)
                    graph[neighbor_idx].add(i)
        
        # Perform transitive closure to find connected components (text lines)
        # This is exactly as described in the paper
        visited = set()
        text_lines = []
        
        def dfs(node, current_line):
            visited.add(node)
            current_line.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_line)
        
        for i in range(len(components)):
            if i not in visited:
                current_line = []
                dfs(i, current_line)
                text_lines.append(current_line)
        
        # Perform linear regression to get more accurate text lines as described in the paper
        refined_lines = []
        for line in text_lines:
            if len(line) < 2:
                refined_lines.append(line)
                continue
                
            # Extract centroids for regression
            centroids = np.array([components[idx].centroid for idx in line])
            x = centroids[:, 0]
            y = centroids[:, 1]
            
            # Perform linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Sort components by x-coordinate for left-to-right reading
            sorted_line = sorted(line, key=lambda idx: components[idx].centroid[0])
            refined_lines.append(sorted_line)
        
        # Sort text lines by y-coordinate (top to bottom)
        # This matches the reading order mentioned in the paper
        refined_lines.sort(key=lambda line: np.mean([components[idx].centroid[1] for idx in line]))
        
        return refined_lines

    def find_accurate_orientation(self, components: List[Component], text_lines: List[List[int]]) -> float:
        """
        Make a more accurate orientation measurement using text lines as described in the paper.
        
        Args:
            components: List of components
            text_lines: List of text lines
            
        Returns:
            Refined orientation estimate in degrees
        """
        if not text_lines:
            return 0
            
        # Only use long text lines for better accuracy as implied in the paper
        long_lines = [line for line in text_lines if len(line) >= 5]
        if not long_lines:
            long_lines = text_lines
            
        # Calculate orientation for each line
        orientations = []
        for line in long_lines:
            if len(line) < 2:
                continue
                
            # Extract centroids
            centroids = [components[idx].centroid for idx in line]
            
            # Calculate orientation via linear regression
            x = [c[0] for c in centroids]
            y = [c[1] for c in centroids]
            
            if len(set(x)) < 2:  # Avoid vertical lines
                continue
                
            # Fit line
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Convert slope to angle
            angle = np.degrees(np.arctan(m)) % 180
            orientations.append(angle)
        
        if not orientations:
            return 0
            
        # Return median orientation for robustness
        return np.median(orientations)

    def find_blocks(self, components: List[Component], text_lines: List[List[int]], 
                  orientation: float, 
                  within_line_spacing: float, 
                  between_line_spacing: float) -> List[List[List[int]]]:
        """
        Group text lines into blocks based on the criteria in the paper:
        1. Parallelness
        2. Perpendicular proximity
        3. Overlap
        4. Parallel proximity
        
        Args:
            components: List of components
            text_lines: List of text lines
            orientation: Refined orientation estimate
            within_line_spacing: Estimated within-line spacing
            between_line_spacing: Estimated between-line spacing
            
        Returns:
            List of blocks, where each block is a list of text lines
        """
        if not text_lines:
            return []
        
        # Define parameters based on spacing values as mentioned in the paper
        perpendicular_threshold = 1.3 * between_line_spacing
        parallel_threshold = 1.5 * within_line_spacing
        
        # Get line bounds
        line_bounds = []
        for line in text_lines:
            if not line:
                line_bounds.append((0, 0, 0, 0))
                continue
                
            min_x = min(components[idx].bbox[0] for idx in line)
            min_y = min(components[idx].bbox[1] for idx in line)
            max_x = max(components[idx].bbox[0] + components[idx].bbox[2] for idx in line)
            max_y = max(components[idx].bbox[1] + components[idx].bbox[3] for idx in line)
            line_bounds.append((min_x, min_y, max_x, max_y))
        
        # Initialize blocks - each text line starts as its own block
        blocks = [[line] for line in text_lines]
        block_bounds = line_bounds.copy()
        
        # Merge blocks according to the criteria described in the paper
        while True:
            merged = False
            
            for i in range(len(blocks)):
                if merged:
                    break
                    
                for j in range(i + 1, len(blocks)):
                    # Get bounds
                    bounds_i = block_bounds[i]
                    bounds_j = block_bounds[j]
                    
                    # Check criteria
                    
                    # 1. Parallelness - assumed for same document, so always true
                    
                    # 2. Perpendicular proximity
                    perp_distance = min(abs(bounds_i[3] - bounds_j[1]), abs(bounds_i[1] - bounds_j[3]))
                    
                    # 3. Overlap - check horizontal overlap
                    x_overlap = min(bounds_i[2], bounds_j[2]) - max(bounds_i[0], bounds_j[0])
                    has_overlap = x_overlap > 0
                    
                    # 4. Parallel proximity - check horizontal distance
                    if bounds_i[2] < bounds_j[0]:  # block i is to the left of block j
                        para_distance = bounds_j[0] - bounds_i[2]
                    elif bounds_j[2] < bounds_i[0]:  # block j is to the left of block i
                        para_distance = bounds_i[0] - bounds_j[2]
                    else:  # Blocks overlap horizontally
                        para_distance = 0
                    
                    # Apply the criteria
                    if perp_distance <= perpendicular_threshold and (has_overlap or para_distance <= parallel_threshold):
                        # Merge blocks
                        blocks[i].extend(blocks[j])
                        blocks.pop(j)
                        
                        # Update bounds
                        min_x = min(bounds_i[0], bounds_j[0])
                        min_y = min(bounds_i[1], bounds_j[1])
                        max_x = max(bounds_i[2], bounds_j[2])
                        max_y = max(bounds_i[3], bounds_j[3])
                        block_bounds[i] = (min_x, min_y, max_x, max_y)
                        block_bounds.pop(j)
                        
                        merged = True
                        break
            
            if not merged:
                break
        
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
        initial_orientation = self.estimate_orientation(neighbors_info)
        
        # Estimate spacing
        within_line_spacing, between_line_spacing = self.estimate_spacing(neighbors_info, initial_orientation)
        
        # Find text lines
        text_lines = self.find_text_lines(components, neighbors_info, initial_orientation)
        
        # Make more accurate orientation measurement using text lines
        final_orientation = self.find_accurate_orientation(components, text_lines)
        
        # Find blocks
        blocks = self.find_blocks(components, text_lines, final_orientation, within_line_spacing, between_line_spacing)
        
        return components, text_lines, final_orientation, blocks, (within_line_spacing, between_line_spacing), initial_orientation

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
        components, text_lines, orientation, blocks, spacings, initial_orientation = docstrum.process(image)
        docstrum.visualize_results(image, components, text_lines, blocks, orientation, spacings, 
                                 args.output_dir, filename)
        
        print(f"Processed {filename}:")
        print(f"- Initial orientation: {initial_orientation:.1f}°")
        print(f"- Final orientation: {orientation:.1f}°")
        print(f"- Within-line spacing: {spacings[0]:.1f} pixels")
        print(f"- Between-line spacing: {spacings[1]:.1f} pixels")
        print(f"- Found {len(components)} components")
        print(f"- Grouped into {len(text_lines)} text lines")
        print(f"- Detected {len(blocks)} text blocks")
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
                components, text_lines, orientation, blocks, spacings, initial_orientation = docstrum.process(image)
                docstrum.visualize_results(image, components, text_lines, blocks, orientation, spacings, 
                                         args.output_dir, base_filename)
                
                print(f"Processed {filename}:")
                print(f"- Initial orientation: {initial_orientation:.1f}°")
                print(f"- Final orientation: {orientation:.1f}°")
                print(f"- Within-line spacing: {spacings[0]:.1f} pixels")
                print(f"- Between-line spacing: {spacings[1]:.1f} pixels")
                print(f"- Found {len(components)} components")
                print(f"- Grouped into {len(text_lines)} text lines")
                print(f"- Detected {len(blocks)} text blocks")
                print(f"- Output saved to {args.output_dir}")
                
                processed += 1
        
        print(f"\nProcessed {processed} images. Output saved to {args.output_dir}")
        
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")

if __name__ == '__main__':
    main()