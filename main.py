import argparse
from os.path import join

import numpy as np
from PIL import Image

from pipeline import (binarize,
                      apply_kfill,
                      find_text_components,
                      filter_component_size,
                      find_nearest_neighbors,
                      estimate_orientation,
                      estimate_spacing,
                      find_text_lines,
                      find_text_blocks)
from visualization import (visualize_text_components,
                           visualize_neighbors,
                           visualize_docstrum,
                           visualize_text_lines,
                           visualize_text_blocks)


def main(img_gray: np.ndarray,
         output_dir: str,
         kfill: bool = True,
         kfill_k: int = 5,
         kfill_m: int = 10,
         nn_k: int = 5,
         angle_tol: float = 30.0,
         ):
    # Binarize
    img_binary = binarize(img_gray)
    Image.fromarray(~img_binary).save(join(output_dir, 'binarized.png'))

    # Apply kFill salt-and-pepper denoising
    img_binary = apply_kfill(img_binary, k=kfill_k, max_iterations=kfill_m)
    Image.fromarray(~img_binary).save(join(output_dir, 'filtered.png'))
    
    # Extract text components and filter by size
    comps_raw = find_text_components(img_binary)
    comps = filter_component_size(comps_raw)
    vis_comps = visualize_text_components(img_binary, comps)
    Image.fromarray(vis_comps).save(join(output_dir, 'components.png'))
    print(f"- Found {len(comps)} text components.")

    # Find nearest neighbors
    idxs_ngbr, vects_ngbr = find_nearest_neighbors(comps, k=nn_k)
    img_neighbors = visualize_neighbors(img_binary, comps, vects_ngbr)
    Image.fromarray(img_neighbors).save(join(output_dir, 'neighbors.png'))

    # Plot docstrum
    visualize_docstrum(vects_ngbr, save_fig=join(output_dir, 'docstrum.png'))

    # Estimate text orientation
    theta = estimate_orientation(vects_ngbr,
        save_hist=join(output_dir, 'orientation.png'))
    print(f"- Estimated text orientation: {theta:.1f}°")
    
    # Estimate text spacing
    spacing_wit, spacing_bet = estimate_spacing(vects_ngbr, theta,
        angle_tolerance=angle_tol,
        save_hist=join(output_dir, 'spacing.png'))
    text_lines, PQs = find_text_lines(comps, idxs_ngbr, vects_ngbr,
                      spacing_bet, spacing_wit, theta)
    print(f"- Estimated within-line spacing: {spacing_wit:.1f}px")
    print(f"- Estimated between-line spacing: {spacing_bet:.1f}px")

    # Group components into text lines
    text_lines, PQs = find_text_lines(comps, idxs_ngbr, vects_ngbr,
                                      spacing_bet, spacing_wit, theta,
                                      angle_tolerance=angle_tol)
    img_lines = visualize_text_lines(img_binary, comps, PQs)
    Image.fromarray(img_lines).save(join(output_dir, 'lines.png'))
    print(f"- Detected {len(text_lines)} text lines.")

    # Group text lines into text blocks
    text_blocks = find_text_blocks(PQs,
                                   da_max=1.5*spacing_wit,
                                   de_max=1.3*spacing_bet,
                                   max_angle=30)
    img_blocks = visualize_text_blocks(img_binary, comps,
                                       text_blocks, text_lines, PQs)
    Image.fromarray(img_blocks).save(join(output_dir, 'blocks.png'))
    print(f"- Detected {len(text_blocks)} text blocks.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', metavar='img-path', type=str,
                        help="Path to the input image file.")
    parser.add_argument('--out-path', type=str,
                        help="Path to directory of outputs.")
    parser.add_argument('--kfill-k', type=int,
                        help="Size of the kFill window.")
    parser.add_argument('--kfill-m', type=int,
                        help="Maximum number of iterations for kFill.")
    parser.add_argument('--nn-k', type=int,
                        help="Number of nearest neighbors.")
    parser.add_argument('--angle-tol', type=float,
                        help="Tolerance for angle estimation, in degrees.")
    args = parser.parse_args()

    img_gray = np.array(Image.open(args.img_path).convert('F'))
    main(img_gray,
         output_dir=args.out_path,
         kfill_k=args.kfill_k,
         kfill_m=args.kfill_m,
         nn_k=args.nn_k,
         angle_tol=args.angle_tol)