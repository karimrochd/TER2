import argparse

import numpy as np
from PIL import Image
from scipy.sparse import csr_array, diags_array, eye_array, kron, vstack
from scipy.sparse.linalg import spsolve
from skimage.filters import threshold_otsu
from skimage.morphology import diamond, dilation


### Poisson editing, following Enric Meinhardt-Llopis @mnhrdt
#
# The image is discretised as the graph (V,E) of the rectangular grid: each
# {interior,edge,corner} pixel is a vertex of degree {4,3,2}. 
#
# The gradient is then a matrix |E| x |V|
# The divergence is minus its transpose, a matrix |V| x |E|
# The laplacian is the divergence of the gradient, a matrix |V| x |V|


def discrete_gradient(h: int, w: int) -> csr_array:
	"""Matrix of the gradient operator for a rectangular domain."""
	x = eye_array(w-1, w, k=1) - eye_array(w-1, w)  # path graph of length W
	y = eye_array(h-1, h, k=1) - eye_array(h-1, h)  # path graph of length H
	p = kron(eye_array(h), x, format='csr')         # H horizontal paths
	q = kron(y, eye_array(w), format='csr')         # W vertical paths
	B = vstack([p, q])                              # union of all paths
	return B


def solve_poisson(f: np.ndarray,
				  g: np.ndarray,
				  m: np.ndarray
				  ) -> np.ndarray:
	"""Solve the Poisson equation: find an image u such that
			Δu = f  where m
	       	 u = g  where not m
	(solution by local discrete method).
	
	Args:
		f: target laplacian, shape (h, w)
		g: boundary condition, shape (h, w)
		m: domain mask, shape (h, w)

	Returns:
		u: solution image, shape (h, w)
	"""
	# flatten the images into vectors
	h, w = f.shape
	f = f.flatten()
	g = g.flatten()
	m = m.flatten() * 1.0

	# state and solve the linear system
	B = discrete_gradient(h, w)            # gradient operator
	L = -B.T @ B                           # laplacian operator
	M = diags_array(m, format='csr')       # mask operator
	I = eye_array(h*w, h*w, format='csr')  # identity operator
	A = (I - M)     - M @ L                # linear system: matrix
	b = (I - M) @ g - M @ f                # linear system: constant terms
	z = spsolve(A, b)                      # linear system: solution
	u = z.reshape(h, w)                    # recover a 2D array from the solution vector
	return u


def edit_poisson(f: np.ndarray,
				 g: np.ndarray,
				 m: np.ndarray
			   	 ) -> np.ndarray:
	"""Poisson-edit an image (source) into another (destination) inside a mask.
	
	Args:
		f: source image, shape (h, w) | (h, w, c)
		g: destination image, shape (h, w) | (h, w, c)
		m: copying mask, shape (h, w)
		
	Returns:
		output image, shape (h, w) | (h, w, c)
	"""
	# if image is color, call the gray-scale version recursively
	if len(f.shape) == 3:
		return np.stack([edit_poisson(f[..., c], g[..., c], m)
			   			 for c in range(f.shape[2])], axis=2)
	
	# flatten the images into vectors
	h, w = f.shape
	f = f.flatten()
	g = g.flatten()
	m = m.flatten() * 1.0

	# build linear operators
	B = discrete_gradient(h, w)  # gradient operator

	# compute gradients of each image
	nf = B @ f                   # gradient of source image
	ng = B @ g                   # gradient of destination image

	# compute the target laplacian x
	dm = (B @ m != 0)               # boundary of the mask (edge mask)
	x = -B.T @ ((1-dm)*nf + dm*ng)  # divergence of the combined gradient

	# recover the image from this gradient
	return solve_poisson(x.reshape(h, w), g, m)


def bg_flatten(img_doc: np.ndarray,
               d: int = 0,
               equalize: bool = True,
               tile_size: int | None = None) -> np.ndarray:
    """Flatten the background of a document image; convert from 8-bit to float.
    
    Args:
        img_doc: 8-bit grayscale document image, shape (h, w).
        d: dilation size for text mask.
        equalize: whether to have consistent output contrast.
        tile_size: if not None, image is processed in tiles of this size.
        
    Returns:
        8-bit image with background flattened to 255, shape (h, w).
    """
    img = img_doc.astype(np.float32)/255
    # Text mask
    thr = threshold_otsu(img)
    mask_txt = dilation((img <= thr), diamond(d, decomposition='sequence'))
    # Tiling
    h, w = img.shape
    t = tile_size
    out = np.zeros((h, w), dtype=np.float32)
    if t is None:
        t = max(h, w)
    for i in range(0, h, t):
        for j in range(0, w, t):
            img_tile = img[i:i+t, j:j+t]
            out_tile = out[i:i+t, j:j+t]
            mask_tile = mask_txt[i:i+t, j:j+t]
            # Poisson editing
            out_tile[:, :] = edit_poisson(f=img_tile,
                                          g=np.ones_like(img_tile),
                                          m=mask_tile)
    # Equalization
    if equalize:
        dec = 1 - np.quantile(out[out<=1-1/255], 0.1)
        out = (out-1)*0.9/dec + 1
    # Clipping
    return np.rint(255*np.clip(out, 0, 1)).astype(np.uint8)


if __name__ == '__main__':
    # Example
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('-d', type=int, default=2)
    parser.add_argument('--equalize', action='store_true')
    args = parser.parse_args()

    # Read
    img = np.array(Image.open(args.img_path).convert('L'))
    # Process
    img_flat = bg_flatten(img, d=args.d, equalize=args.equalize, tile_size=2000)
    # Write
    Image.fromarray(img_flat).save(args.out_path)
