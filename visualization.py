import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipeline import Component


def visualize_text_components(img_binary: np.ndarray, components: list[Component]):
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


def visualize_neighbors(img_binary: np.ndarray,
                        components: list[Component], 
                        vectors: list[list[tuple[int, float, float]]]):
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


def visualize_docstrum(vectors, save_fig: str):
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
    
    plt.savefig(save_fig, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_text_lines(img_binary: np.ndarray,
                    components: list[Component], 
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
                 color, 2, cv2.LINE_AA)
    
    # Draw component centroids on top
    for comp in components:
        x, y = tuple(map(int, np.rint(comp.centroid)))
        img_vis[y, x] = (255, 0, 0)
    
    return img_vis


def visualize_text_blocks(img_binary: np.ndarray,
                          components: list[Component], 
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
    for idx_block, block in enumerate(blocks):
        color = tuple(int(np.rint(c*255))
                          for c in plt.cm.tab10((idx_block/10)%1)[:3])

        # Draw block bounding box
        xywh = np.stack([components[j].bbox for i in block for j in lines[i]],
                        axis=0)
        ltrb = xywh @ (np.diag([1, 1, 1, 1]) + np.diag([1, 1], k=2))
        l, t = ltrb[:, 0].min(), ltrb[:, 1].min()
        r, b = ltrb[:, 2].max(), ltrb[:, 3].max()
        cv2.rectangle(img_vis, (l, t), (r, b), color, 2)

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