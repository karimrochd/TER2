"""
Main entry point for Docstrum document layout analysis.

This module handles command-line arguments, orchestrates the analysis
pipeline, and manages batch processing of images.
"""

import argparse
import cv2
import sys
import os
import logging
import numpy as np
from typing import List, Tuple, Optional

from segmentation import Docstrum, Component, calculate_vertical_threshold
from preprocess import rotate_image
from visualize import (
    visualize_preprocessing, 
    visualize_components, 
    visualize_neighbors,
    visualize_orientation_histogram, 
    visualize_docstrum, 
    visualize_text_lines,
    visualize_initial_blocks, 
    visualize_final_blocks, 
    generate_distinct_colors
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_processing_details(filename: str, components: List[Component], 
                         text_lines: List[List[int]], blocks: List[List[List[int]]], 
                         orientation: float, just_lines: bool, 
                         output_dir: str, log_file: str):
    """
    Log processing details to a file.
    
    Args:
        filename: Name of the processed file
        components: List of detected components
        text_lines: List of detected text lines
        blocks: List of detected blocks
        orientation: Detected orientation in degrees
        just_lines: Boolean indicating merge mode
        output_dir: Output directory path
        log_file: Path to the log file
    """
    log_entry = f"""
Processed {filename}:
- Found {len(components)} components
- Grouped into {len(text_lines)} text lines
- Detected {len(blocks)} text blocks
- Estimated orientation: {orientation:.1f} degrees
- Merge mode: {'line-only' if just_lines else 'lines and vertical'}
- Saved visualization to {output_dir}
----------------------------------------
"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def process_and_save_visualization(
    image: np.ndarray, 
    output_dir: str, 
    filename: str,
    docstrum: Docstrum,
    args: argparse.Namespace
) -> Tuple[List[Component], List[List[int]], float, List[List[List[int]]]]:
    """
    Process an image through the complete Docstrum pipeline.
    
    Args:
        image: Input grayscale image
        output_dir: Output directory for visualizations
        filename: Base filename for outputs
        docstrum: Configured Docstrum instance
        args: Command-line arguments
        
    Returns:
        Tuple of (components, text_lines, orientation, blocks)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Preprocessing
    logger.info(f"Preprocessing {filename}...")
    binary = docstrum.preprocess(
        image,
        small_component_threshold=args.small_component_threshold,
        binarization_threshold=args.binarization_threshold,
        kfill_threshold=args.kfill_threshold,
        filter_type=args.filter_type,
        kfill_iterations=args.kfill_iterations
    )
    visualize_preprocessing(image, binary, output_dir, filename)
    
    # Step 2: Find components
    logger.info(f"Finding connected components in {filename}...")
    components = docstrum.find_connected_components(binary, args.big_component_threshold)
    visualize_components(image, components, output_dir, filename)
    
    # Step 3: Find nearest neighbors
    logger.info(f"Analyzing nearest neighbors in {filename}...")
    neighbors_info = docstrum.find_nearest_neighbors(components)
    visualize_neighbors(image, components, neighbors_info, output_dir, filename)
    
    # Step 4: Estimate orientation
    logger.info(f"Estimating orientation for {filename}...")
    orientation = docstrum.estimate_orientation(args.smoothing_arg, neighbors_info)
    visualize_orientation_histogram(neighbors_info, orientation, output_dir, 
                                  filename, args.smoothing_arg)
    visualize_docstrum(components, neighbors_info, output_dir, filename)
    
    logger.info(f"Estimated orientation: {orientation:.2f} degrees")
    
    # Step 5: Process with rotation if needed
    if abs(orientation) > 0.5:  # Significant rotation detected
        logger.info(f"Rotating {filename} by {orientation:.2f} degrees...")
        
        # Rotate images
        rotated_image = rotate_image(image, orientation, 255)
        rotated_binary = rotate_image(binary, orientation, 0)
        
        # Save rotated preprocessing
        visualize_preprocessing(rotated_image, rotated_binary, output_dir, 
                              f"{filename}_rotated")
        
        # Re-find components on rotated image
        rotated_components = docstrum.find_connected_components(
            rotated_binary, args.big_component_threshold
        )
        visualize_components(rotated_image, rotated_components, output_dir, 
                           f"{filename}_rotated")
        
        # Re-analyze with rotated data
        rotated_neighbors_info = docstrum.find_nearest_neighbors(rotated_components)
        rotated_orientation = docstrum.estimate_orientation(
            args.smoothing_arg, rotated_neighbors_info
        )
        
        logger.info(f"Orientation after rotation: {rotated_orientation:.2f} degrees")
        
        # Find text lines
        rotated_text_lines = docstrum.find_text_lines(
            rotated_components, rotated_neighbors_info,
            rotated_orientation, spacing_factor=args.spacing_factor
        )
        visualize_text_lines(rotated_image, rotated_components, rotated_text_lines,
                           output_dir, f"{filename}_rotated")
        
        # Calculate vertical threshold if needed
        vertical_threshold = args.vertical_distance_threshold
        if vertical_threshold == -1:
            vertical_threshold = calculate_vertical_threshold(
                rotated_text_lines, rotated_components
            )
            logger.info(f"Auto-calculated vertical threshold: {vertical_threshold:.2f}")
        
        # Find and merge blocks
        rotated_initial_blocks = docstrum.find_blocks(
            rotated_components, rotated_text_lines
        )
        visualize_initial_blocks(rotated_image, rotated_components, 
                               rotated_initial_blocks, output_dir, f"{filename}_rotated")
        
        rotated_merged_blocks = docstrum.merge_overlapping_blocks(
            rotated_components, rotated_initial_blocks,
            horizontal_distance_threshold=args.horizontal_distance_threshold,
            vertical_distance_threshold=vertical_threshold,
            just_lines=args.just_lines,
            block_overlap_threshold=args.block_overlap_threshold
        )
        
        # Create final visualizations
        visualize_final_blocks(rotated_image, rotated_components, 
                             rotated_merged_blocks, output_dir, 
                             f"{filename}_rotated", "Rotated Final Blocks")
        
        # Rotate visualization back to original orientation
        rotated_vis_image = create_blocks_visualization(
            rotated_image, rotated_components, rotated_merged_blocks
        )
        final_vis_image = rotate_image(rotated_vis_image, -orientation)
        
        final_output_path = os.path.join(output_dir, f'{filename}_final_blocks.png')
        cv2.imwrite(final_output_path, cv2.cvtColor(final_vis_image, cv2.COLOR_RGB2BGR))
        
        # Log results
        log_processing_details(filename, rotated_components, rotated_text_lines,
                             rotated_merged_blocks, orientation, args.just_lines,
                             output_dir, args.log_file)
        
        return rotated_components, rotated_text_lines, orientation, rotated_merged_blocks
    
    else:
        # Process without rotation
        logger.info(f"Processing {filename} without rotation...")
        
        text_lines = docstrum.find_text_lines(
            components, neighbors_info, orientation, spacing_factor=args.spacing_factor
        )
        visualize_text_lines(image, components, text_lines, output_dir, filename)
        
        # Calculate vertical threshold if needed
        vertical_threshold = args.vertical_distance_threshold
        if vertical_threshold == -1:
            vertical_threshold = calculate_vertical_threshold(text_lines, components)
            logger.info(f"Auto-calculated vertical threshold: {vertical_threshold:.2f}")
        
        # Find and merge blocks
        initial_blocks = docstrum.find_blocks(components, text_lines)
        visualize_initial_blocks(image, components, initial_blocks, output_dir, filename)
        
        merged_blocks = docstrum.merge_overlapping_blocks(
            components, initial_blocks,
            horizontal_distance_threshold=args.horizontal_distance_threshold,
            vertical_distance_threshold=vertical_threshold,
            just_lines=args.just_lines,
            block_overlap_threshold=args.block_overlap_threshold
        )
        
        # Create final visualization
        visualize_final_blocks(image, components, merged_blocks, output_dir, filename)
        
        # Log results
        log_processing_details(filename, components, text_lines, merged_blocks,
                             orientation, args.just_lines, output_dir, args.log_file)
        
        return components, text_lines, orientation, merged_blocks


def create_blocks_visualization(image: np.ndarray, components: List[Component],
                              blocks: List[List[List[int]]]) -> np.ndarray:
    """
    Create a visualization image with colored blocks.
    
    Args:
        image: Original grayscale image
        components: List of components with bbox in (x1, y1, x2, y2) format
        blocks: List of blocks (each block contains lines of component indices)
        
    Returns:
        RGB visualization image with colored block boundaries
    """
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    colors = generate_distinct_colors(len(blocks))
    
    for block_idx, block in enumerate(blocks):
        block_components = [comp_idx for line in block for comp_idx in line]
        if not block_components:
            continue
        
        min_x = min(components[idx].bbox[0] for idx in block_components)
        min_y = min(components[idx].bbox[1] for idx in block_components)
        max_x = max(components[idx].bbox[2] for idx in block_components)
        max_y = max(components[idx].bbox[3] for idx in block_components)
        
        color = colors[block_idx % len(colors)].tolist()
        padding = 3
        cv2.rectangle(vis_image, 
                    (min_x - padding, min_y - padding),
                    (max_x + padding, max_y + padding),
                    color, 2)
    
    return vis_image


def process_single_image(image_path: str, args: argparse.Namespace, 
                       docstrum: Docstrum) -> Optional[Tuple]:
    """
    Process a single image file.
    
    Args:
        image_path: Path to the image
        args: Command-line arguments
        docstrum: Configured Docstrum instance
        
    Returns:
        Processing results or None if error
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Could not load image {image_path}")
        return None
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        results = process_and_save_visualization(
            image, args.output_dir, filename, docstrum, args
        )
        
        logger.info(f"\nSuccessfully processed {filename}:")
        logger.info(f"- Found {len(results[0])} components")
        logger.info(f"- Grouped into {len(results[1])} text lines")
        logger.info(f"- Detected {len(results[3])} text blocks")
        logger.info(f"- Orientation: {results[2]:.1f} degrees")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write(f"Error processing {filename}: {str(e)}\n")
        return None


def process_directory(input_dir: str, args: argparse.Namespace, 
                     docstrum: Docstrum) -> Tuple[int, int]:
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing images
        args: Command-line arguments
        docstrum: Configured Docstrum instance
        
    Returns:
        Tuple of (processed_count, error_count)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    processed = 0
    errors = 0
    
    for filename in os.listdir(input_dir):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_path = os.path.join(input_dir, filename)
            result = process_single_image(image_path, args, docstrum)
            
            if result is not None:
                processed += 1
            else:
                errors += 1
    
    return processed, errors


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Run Docstrum page layout analysis on document images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single image:
    python main.py image.jpg --output_dir results
    
  Process a directory:
    python main.py images/ --output_dir results --just_lines
    
  With custom parameters:
    python main.py image.jpg --k_nearest 7 --angle_threshold 3.0
        """
    )
    
    # Required arguments
    parser.add_argument('input_path', type=str, 
                       help='Path to input image or directory')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save output visualizations (default: output)')
    parser.add_argument('--log_file', type=str, default='processing_log.log',
                       help='Path to the log file (default: processing_log.log)')
    
    # Docstrum parameters
    parser.add_argument('--k_nearest', type=int, default=5, 
                       help='Number of nearest neighbors (default: 5)')
    parser.add_argument('--angle_threshold', type=float, default=5.0, 
                       help='Angle threshold in degrees (default: 5.0)')
    parser.add_argument('--spacing_factor', type=float, default=1.2,  
                       help='Factor for max allowed gap (default: 1.2)')
    
    # Preprocessing parameters
    parser.add_argument('--binarization_threshold', type=int, default=-1,
                       help='Binarization threshold, -1 for Otsu (default: -1)')
    parser.add_argument('--small_component_threshold', type=float, default=0.05,
                       help='Small component threshold (default: 0.05)')
    parser.add_argument('--big_component_threshold', type=int, default=-1,
                       help='Large component threshold, -1 to disable (default: -1)')
    parser.add_argument('--kfill_threshold', type=int, default=5,
                       help='kFill window size (default: 5)')
    parser.add_argument('--filter_type', type=int, default=2, choices=[0, 1, 2],
                       help='Filter type: 0=kfill, 1=size, 2=both (default: 2)')
    parser.add_argument('--kfill_iterations', type=int, default=10,
                       help='kFill iterations (default: 10)')
    
    # Merging parameters
    parser.add_argument('--horizontal_distance_threshold', type=float, default=12.0,
                       help='Max horizontal distance for merging (default: 12.0)')
    parser.add_argument('--vertical_distance_threshold', type=float, default=-1.0,
                       help='Max vertical distance, -1 for auto (default: -1.0)')
    parser.add_argument('--block_overlap_threshold', type=float, default=0.9,
                       help='Block overlap threshold (default: 0.9)')
    parser.add_argument('--just_lines', action='store_true', default=False,
                       help='Only merge blocks in the same line')
    
    # Other parameters
    parser.add_argument('--smoothing_arg', type=int, default=91,
                       help='Smoothing window for orientation (default: 91)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.binarization_threshold < -1:
        logger.error("Binarization threshold must be -1 or non-negative")
        sys.exit(1)
    
    # Initialize Docstrum
    docstrum = Docstrum(k_nearest=args.k_nearest, 
                       angle_threshold=args.angle_threshold)
    
    # Process input
    if os.path.isfile(args.input_path):
        logger.info(f"Processing single image: {args.input_path}")
        result = process_single_image(args.input_path, args, docstrum)
        
        if result is None:
            sys.exit(1)
            
    elif os.path.isdir(args.input_path):
        logger.info(f"Processing directory: {args.input_path}")
        processed, errors = process_directory(args.input_path, args, docstrum)
        
        # Log summary
        summary = f"""
Processing complete:
- Successfully processed: {processed} images
- Errors: {errors} images
- Output saved to: {args.output_dir}
========================================
"""
        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(summary)
        
    else:
        logger.error(f"{args.input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()