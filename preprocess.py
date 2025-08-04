"""
Image preprocessing utilities for document layout analysis.

This module provides functions for image binarization, noise reduction,
and geometric transformations.
"""

import cv2
import numpy as np
from typing import Tuple, Union


def kfill(binary_image: np.ndarray, k: int = 5, max_iterations: int = 10) -> np.ndarray:
    """
    Implement the kFill filter for noise reduction in binary document images.
    
    The kFill algorithm fills small holes and removes small noise in binary images
    by analyzing local neighborhoods and applying morphological-like operations.
    
    Args:
        binary_image: Binary image (1 for foreground, 0 for background)
        k: Window size parameter (must be odd)
        max_iterations: Maximum number of iterations to perform
        
    Returns:
        Filtered binary image
        
    Raises:
        ValueError: If k is even (must be odd for symmetric window)
    """
    # Ensure k is odd
    if k % 2 == 0:
        k = k + 1
    
    filtered_image = binary_image.copy()
    iteration = 0
    changes_made = True
    
    while changes_made and iteration < max_iterations:
        changes_made = False
        iteration += 1
        
        # Perform ON-fill and OFF-fill sub-iterations
        for fill_value in [1, 0]:  # 1 for ON-fill, 0 for OFF-fill
            height, width = filtered_image.shape
            temp_image = filtered_image.copy()
            
            # Process each pixel
            for y in range(k//2, height - k//2):
                for x in range(k//2, width - k//2):
                    # Extract window
                    window = filtered_image[y - k//2 : y + k//2 + 1, 
                                          x - k//2 : x + k//2 + 1]
                    
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
                        n = np.sum(neighborhood == 1)
                    else:
                        n = np.sum(neighborhood == 0)
                    
                    # Calculate c (number of connected groups in neighborhood)
                    expanded_neighborhood = np.concatenate([neighborhood, neighborhood[0:1]])
                    c = 0
                    for i in range(len(neighborhood)):
                        if expanded_neighborhood[i] != expanded_neighborhood[i+1]:
                            c += 1
                    c = c // 2  # Each transition is counted twice
                    
                    # Calculate r (number of corner pixels that are ON or OFF)
                    corners = [window[0, 0], window[0, -1], window[-1, 0], window[-1, -1]]
                    if fill_value == 1:
                        r = sum(1 for corner in corners if corner == 1)
                    else:
                        r = sum(1 for corner in corners if corner == 0)
                    
                    # Apply kFill condition
                    if (c == 1) and ((n > 3*k - 4) or ((n == 3*k - 4) and (r == 2))):
                        temp_image[y - k//2 + 1 : y + k//2, 
                                 x - k//2 + 1 : x + k//2] = fill_value
                        changes_made = True
            
            filtered_image = temp_image.copy()
    
    print(f"kFill completed after {iteration} iterations.")
    return filtered_image


def binarize_image(image: np.ndarray, threshold: int = -1) -> np.ndarray:
    """
    Binarize a grayscale image using either Otsu's method or a fixed threshold.
    
    Args:
        image: Input grayscale image
        threshold: Threshold value (-1 for Otsu's automatic thresholding)
        
    Returns:
        Binary image (0 for background, 1 for foreground)
    """
    if threshold == -1:
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Convert to 0 and 1 (text as 1)
    binary = (binary == 0).astype(np.uint8)
    
    return binary


def remove_small_components(binary_image: np.ndarray, 
                          small_component_threshold: float = 0.05,
                          kfill_threshold: int = 5, 
                          filter_type: int = 2,
                          kfill_iterations: int = 10) -> np.ndarray:
    """
    Remove connected components smaller than a threshold based on component size distribution.
    
    This function identifies the most common component size and removes components
    that are significantly smaller, which are likely to be noise.
    
    Args:
        binary_image: Binary image (1 for foreground, 0 for background)
        small_component_threshold: Threshold multiplier for peak component area
        kfill_threshold: Threshold for kFill filter
        filter_type: Type of filtering (0: kFill only, 1: size only, 2: both)
        kfill_iterations: Maximum iterations for kFill
        
    Returns:
        Binary image with small components removed
    """
    # Apply kFill if requested
    if filter_type == 0 or filter_type == 2:
        binary_image = kfill(binary_image, k=kfill_threshold, 
                           max_iterations=kfill_iterations)
    
    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    
    output_image = np.zeros_like(binary_image, dtype=np.uint8)
    
    # Calculate component size threshold
    if num_labels > 1:
        # Use bounding box areas
        bbox_areas = [stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT] 
                     for i in range(1, num_labels)]
        
        # Find peak in histogram (most common size)
        hist, bin_edges = np.histogram(bbox_areas, bins='auto')
        peak_bin_index = np.argmax(hist)
        peak_area = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
        
        min_area_threshold = peak_area * small_component_threshold
        print(f"Component area threshold: {min_area_threshold:.2f}")
    else:
        min_area_threshold = small_component_threshold
        print("No components found, using default threshold")
    
    # Keep components larger than threshold
    for i in range(1, num_labels):  # Skip background
        component_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
        if filter_type == 0 or component_area >= min_area_threshold:
            output_image[labels == i] = 1
    
    return output_image


def rotate_image(image: np.ndarray, angle_degrees: float, 
                background_value: Union[int, Tuple[int, int, int]] = 255) -> np.ndarray:
    """
    Rotate an image by the specified angle while preserving all content.
    
    Args:
        image: Image to rotate (grayscale or color)
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)
        background_value: Value to fill background after rotation
        
    Returns:
        Rotated image with adjusted canvas size
    """
    # Handle color images
    if len(image.shape) > 2 and image.shape[2] == 3:
        background_value = (background_value, background_value, background_value)
    
    # Calculate image center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Calculate new dimensions to fit rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new size
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, 
                           borderValue=background_value)
    
    return rotated