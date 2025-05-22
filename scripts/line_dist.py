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
import numpy as np


# %% [markdown]
# # Functions

# %%
def calculate_A_coordinates(line1, line2):

    slope1, intercept1, x_min1, x_max1 = line1
    slope2, intercept2, x_min2, x_max2 = line2

    y_min1 = slope1 * x_min1 + intercept1
    y_max1 = slope1 * x_max1 + intercept1

    y_min2 = slope2 * x_min2 + intercept2
    y_max2 = slope2 * x_max2 + intercept2


    delta_x1 = x_max1 - x_min1
    delta_y1 = y_max1 - y_min1

    delta_x2 = x_max2 - x_min2
    delta_y2 = y_max2 - y_min2

    if delta_x1 != 0:
        A = x_min1 * delta_x1 * delta_x2
        B = x_min2 * delta_y1 * delta_y2
        C = delta_x2 * delta_y1 * (y_min1 - y_min2)
        D = delta_y1 * delta_y2 + delta_x1 * delta_x2
        x_A2 = (A + B + C) / D
        y_A2 = slope2 * x_A2 + intercept2

        E = x_max1 * delta_x1 * delta_x2
        F = x_max2 * delta_y1 * delta_y2
        G = delta_x2 * delta_y1 * (y_max1 - y_max2)
        x_B2 = (E + F + G) / D
        y_B2 = slope2 * x_B2 + intercept2

    else:
        A = y_min1 * delta_y1 * delta_y2
        B = y_min2 * delta_x1 * delta_x2
        C = delta_y2 * delta_x1 * (x_min1 - x_min2)
        D = delta_x1 * delta_x2 + delta_y1 * delta_y2
        y_A2 = (A + B + C) / D
        x_A2 = (y_A2 - y_min2) * (delta_x2 / delta_y2) + x_min2

        E = y_max1 * delta_y1 * delta_y2
        F = y_max2 * delta_x1 * delta_x2
        G = delta_y2 * delta_x1 * (x_max1 - x_max2)
        y_B2 = (E + F + G) / D
        x_B2 = (y_B2 - y_max2) * (delta_x2 / delta_y2) + x_max2

    return x_A2, y_A2, x_B2, y_B2


def calculate_parallel_distance(line1, line2):
    """
    Calculate the parallel distance between two non-overlapping text lines
    
    Args:
        line1: Tuple of (slope, intercept, x_min, x_max) for the first line
        line2: Tuple of (slope, intercept, x_min, x_max) for the second line
        
    Returns:
        Parallel distance between the two lines (minimum distance between endpoints)
    """
    slope1, intercept1, x_min1, x_max1 = line1
    slope2, intercept2, x_min2, x_max2 = line2

    y_min1 = slope1 * x_min1 + intercept1
    y_max1 = slope1 * x_max1 + intercept1

    y_min2 = slope2 * x_min2 + intercept2
    y_max2 = slope2 * x_max2 + intercept2



    x_A2, y_A2, x_B2, y_B2 = calculate_A_coordinates(line1, line2)

    list_coord = [(x_min2, y_min2), (x_max2, y_max2), (x_A2, y_A2), (x_B2, y_B2)]
    if (x_max2 - x_min2) != 0:
        list_coord.sort(key=lambda x: x[0])
    else:
        list_coord.sort(key=lambda x: x[1])

    (x_C2, y_C2), (x_D2, y_D2) = list_coord[1], list_coord[2]
    
    p2 = np.sqrt((y_D2 - y_C2)**2 + (x_D2 - x_C2)**2)    

    # These middle points are contained within both segments if they are overlapped, or they define a segment between them if they are not overlapped.
    x_min2, x_max2 = min(x_min2, x_max2), max(x_min2, x_max2)
    x_A2, x_B2 = min(x_A2, x_B2), max(x_A2, x_B2)
    if (x_A2 >= x_min2 and x_A2 <= x_max2) or (x_B2 >= x_min2 and x_B2 <= x_max2) or (x_min2 >= x_A2 and x_min2<= x_B2) or (x_max2 >= x_A2 and x_max2<= x_B2) :
        overlap = True
    else :
        overlap = False

    if overlap : 
        parallel_distance = p2
    
    else :
        parallel_distance = -1 * p2

    return parallel_distance


def calculate_perpendicular_distance(line1, line2):
    """
    Calculate the perpendicular distance between two non-overlapping text lines
    
    Args:
        line1: Tuple of (slope, intercept, x_min, x_max) for the first line
        line2: Tuple of (slope, intercept, x_min, x_max) for the second line
        
    Returns:
        Perpendicular distance between the two lines (minimum distance between endpoints)
    """
    slope1, intercept1, x_min1, x_max1 = line1
    slope2, intercept2, x_min2, x_max2 = line2

    y_min1 = slope1 * x_min1 + intercept1
    y_max1 = slope1 * x_max1 + intercept1

    y_min2 = slope2 * x_min2 + intercept2
    y_max2 = slope2 * x_max2 + intercept2



    x_A2, y_A2, x_B2, y_B2 = calculate_A_coordinates(line1, line2)

    list_coord = [(x_min2, y_min2), (x_max2, y_max2), (x_A2, y_A2), (x_B2, y_B2)]
    if (x_max2 - x_min2) != 0:
        list_coord.sort(key=lambda x: x[0])
    else:
        list_coord.sort(key=lambda x: x[1])

    (x_C2, y_C2), (x_D2, y_D2) = list_coord[1], list_coord[2]

    x_M, y_M = (x_C2 + x_D2) / 2, (y_C2 + y_D2) / 2

    delta_x1 = x_max1 - x_min1
    delta_y1 = y_max1 - y_min1

    if delta_x1 == 0:
        return np.abs(x_M - x_min1)
    elif delta_y1 == 0:
        return np.abs(y_M - y_min1)
    else:
        num = (x_M-x_min1) - (y_M-y_min1)*delta_x1/delta_y1
        den = np.sqrt(1 + (delta_x1/delta_y1)**2)
        return np.abs(num / den)


# %%
def calculate_line_distances(line1: np.ndarray,
                             line2: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    # Prepare data
    lines = np.stack([line1, line2], axis=-2) # shape (..., 2, 4)
    slopes = lines[..., None, :, 0] # shape (..., 1, 2)
    intercepts = lines[..., None, :, 1] # shape (..., 1, 2)
    x_mins = lines[..., None, :, 2] # shape (..., 1, 2)
    x_maxs = lines[..., None, :, 3] # shape (..., 1, 2)

    # Matrix (P1, P2, Q1, Q2), shape (..., 2, 4)
    PPQQ = np.block([[x_mins, x_maxs],
                     [slopes*x_mins+intercepts, slopes*x_maxs+intercepts]])
    
    # Rotation matrix of angle -theta1
    delta1 = PPQQ[..., 2] - PPQQ[..., 0] # Q1-P1, shape (..., 2)
    n1 = delta1 / np.linalg.norm(delta1, axis=-1, keepdims=True) # [cos, sin], shape (..., 2)
    Mrot = np.stack([n1, n1[..., ::-1]*[-1, 1]], axis=-2) # [[cos, sin], [-sin, cos]], shape (..., 2, 2)
    # Rotate points
    PPQQ = Mrot @ PPQQ

    # cos(theta2-theta1)
    delta2 = PPQQ[..., 3] - PPQQ[..., 1] # Q2-P2 after rotation, shape (..., 2)
    cos_t12 = delta2[..., 0] / np.linalg.norm(delta2, axis=-1) # shape (...)

    # Order A, B, C, D
    order = np.argsort(PPQQ[..., 0, :], axis=-1) # shape (..., 4)

    # Parallel distance
    inds = np.indices(PPQQ.shape[:-2])
    da = (PPQQ[*inds, 0, order[*inds, 2]] - PPQQ[*inds, 0, order[*inds, 1]]) / np.abs(cos_t12) # (xC-xB)/cos(theta2-theta1), shape (...)
    # Overlap condition
    xLR = np.sort(PPQQ[..., 0, [0, 2]], axis=-1) # (xP1, xQ1) in order, shape (..., 2)
    cond1 = xLR[..., 1] == PPQQ[*inds, 0, order[*inds, 1]] # xR1 == xB
    cond2 = xLR[..., 0] == PPQQ[*inds, 0, order[*inds, 2]] # xL1 == xC
    da *= np.where(cond1 | cond2, -1, 1)

    # Perpendicular distance
    xM = (PPQQ[*inds, 0, order[*inds, 1]] + PPQQ[*inds, 0, order[*inds, 2]]) / 2 # (xB+xC)/2, shape (...)
    alpha = (xM - PPQQ[..., 0, 1]) / (PPQQ[..., 0, 3] - PPQQ[..., 0, 1]) # (xM-xP2)/(xQ2-xP2), shape (...)
    # alpha = 1-alpha
    de = (1-alpha)*PPQQ[..., 1, 1] + alpha*PPQQ[..., 1, 3] - PPQQ[..., 1, 0] # (1-alpha)*yP2 + alpha*yQ2 - yP1, shape (...)
    de = np.abs(de)
    
    return da, de


# %% [markdown]
# # Test difference

# %%
# Generate random data

rng = np.random.default_rng(0)

N = 100000
shape = (N,)

slopes = np.tan(rng.uniform(0, np.pi/2, size=shape+(2,)))
intercepts = 100*rng.normal(size=shape+(2,))
x_mins = 100*rng.normal(size=shape+(2,))
x_maxs = 100*rng.normal(size=shape+(2,))

slopes = slopes[..., None, :]
intercepts = intercepts[..., None, :]
x_mins = x_mins[..., None, :]
x_maxs = x_maxs[..., None, :]

line1, line2 = np.moveaxis(np.concatenate([slopes, intercepts, x_mins, x_maxs],
                                          axis=-2), -1, 0) # shape (..., 4)

# Calculate distances using original paper's equations

das_old, des_old = np.zeros((2, N))
for k in range(N):
    das_old[k] = calculate_parallel_distance(line1[k], line2[k])
    des_old[k] = calculate_perpendicular_distance(line1[k], line2[k])

# Calculate distances using new equations

das_new, des_new = calculate_line_distances(line1, line2)

# Check if the results are the same

print(np.abs((das_old - das_new)/das_old).max())
print(np.abs((des_old - des_new)/des_old).max())
