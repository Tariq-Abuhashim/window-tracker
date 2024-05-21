
def options():
    """
    Returns default options for functions 'lmatch_generate' and 'lmatch_resolve'.
    Any option can be changed, e.g., 'opt["Calibration"] = 3'.

    Options affecting function 'lmatch_generate' only:
    - 'DispRange': displacement search range along epipolars (default: float('inf') pixels).
    - 'MaxViewDistance': the maximal view distance between query view and the nearest view (default: float('inf')).

    Options affecting functions 'lmatch_generate' and 'lmatch_resolve' with 'Merging = 1':
    - 'Calibration': calibration level (1 = projective, 2 = affine, 3 = metric, default: 3).
    - 'NCCWindow': parameters of correlation window used to compute similarity of image line segments.
    - 'NCCThreshold': thresholds on NCC to accept the match.
    - 'ReprojResidual': reprojection residual threshold for linear/nonlinear estimation (default: [5, 1.5] pixels).

    Options affecting function 'lmatch_resolve' only:
    - 'Ordering': 0 = no ordering constraint, 1 = ord. constraint (slow but accurate), 2 = ord. constraint fast but less accurate (default: 0).
    - 'Merging': 0 = merging not applied, 1 = merging applied (default: 0).
    - 'MergeResidual': max. residual for merging fragmented line segments (default: 1).
    """

    opt = {
        'DispRange': float('inf'),
        'MaxViewDistance': float('inf'),
        'Calibration': 3,
        'ReprojResidual': [5, 1.5],
        'NCCWindow': [6, 7, 3, 1],
        'NCCThreshold': [0.6, 10],
        'Ordering': False,  # logical(0) in MATLAB is False in Python
        'Merging': False,   # logical(0) in MATLAB is False in Python
        'MergeResidual': 1
    }

    return opt

# Example usage
# options = lmatch.options()
# # Now options can be accessed, for example:
# print(options['Calibration'])


def orient_lines(lines, image):
    """
    Orients line segments in 'lines' based on the intensity of the 'image'.
    Ensures that the lighter part of the image is on the right of the line segment.
    
    :param lines: A list of dictionaries, each representing a line segment with keys 'u' and 'v'.
    :param image: The image array.
    :return: The oriented line segments.
    """
    h = vgg.gauss_mask(1, 1, (-3, 3))

    for line in lines:
        u = np.array(line['u']).reshape(2, 1)
        v = np.array(line['v']).reshape(2, 1)
        if vgg.conv_lineseg(image, h, u, v) < 0:
            line['u'], line['v'] = line['v'], line['u']

    return lines

# Usage example:
# lines = [{'u': np.array([x1, y1]), 'v': np.array([x2, y2])}, ...]  # Define your line segments here
# image = cv2.imread('image_path', cv2.IMREAD_GRAYSCALE)  # Load your image in grayscale
# oriented_lines = lmatch_lineseg_orient(lines, image)

def extreme_view(K, k, sgn, D):
    """
    Finds the nearest or farthest view of view set K to view k.
    :param K: Iterable of view indices, representing a set of views.
    :param k: The reference view index.
    :param sgn: +1 for nearest view, -1 for farthest view.
    :param D: 2D numpy array, distance table of pairs of views.
    :return: A tuple (Dnear, k0) where Dnear is the distance of the nearest/farthest view, and k0 is its index.
    """
    Dnear = float('inf') * sgn
    k0 = None

    for l in K:
        d = D[l, k]
        if d * sgn < Dnear * sgn:
            k0 = l
            Dnear = d

    return Dnear, k0

# Example usage
# D = np.random.rand(5, 5)  # Replace with your pairwise view distance matrix
# K = [0, 1, 2, 3]  # Set of view indices
# k = 4  # Reference view index
# nearest_distance, nearest_index = extreme_view(K, k, 1, D)
# farthest_distance, farthest_index = extreme_view(K, k, -1, D)

def setminus(a, b):
    """
    Set-theoretical subtraction of index sets a, b.
    Returns elements in a that are not in b.
    """
    return [item for item in a if item not in b]

# Example usage
# a = [1, 2, 3, 4, 5]
# b = [2, 4]
# c = setminus(a, b)  # c will be [1, 3, 5]

def find_basepairs(D):
    K = D.shape[0]

    # find two closest views
    d = np.zeros(K)
    i = np.zeros(K, dtype=int)
    for k in range(K):
        d[k], i[k] = extreme_view(setminus(list(range(K)), [k]), k, 1, D)
    k = np.argmin(d)
    Ki = [[k], [i[k]]]

    while True:
        # k := farthest view from existing set Ki
        d = {}
        for k in setminus(list(range(K)), Ki[0] + Ki[1]):
            d[k], i[k] = extreme_view(Ki[0] + Ki[1], k, 1, D)
        if not d:
            break
        k = max(d, key=d.get)

        # find nearest view to k
        nKi = setminus(list(range(K)), Ki[0] + Ki[1] + [k])
        if not nKi:
            break
        _, i_k = extreme_view(nKi, k, 1, D)
        Ki[0].append(k)
        Ki[1].append(i_k)

    return np.array(Ki)

# Example usage
# D = np.random.rand(5, 5)  # Replace with your pairwise view distance matrix
# basepairs = lmatch_find_basepairs(D)

def print_stat(li):
    K = li.shape[0]
    sli = np.sum(li > 0, axis=1)
    n = np.zeros(K + 1, dtype=int)

    for k in range(2, K + 1):
        n[k] = np.count_nonzero(sli == k)

    print('(', end='')
    for k in np.where(n > 0)[0]:
        if k >= 2:
            print(f'{k}:{n[k]} ', end='')
    print(f'any:{np.sum(n)})', end='')

# Example usage
# li = np.array(...)  # Some numpy array representing match data
# lmatch_print_stat(li)

def remove_overlaps(M):
    print('Removing overlaps .. ', end='')
    r = []  # Matches to be removed

    for i in range(M['li'].shape[1]):
        ki = np.where(M['li'][:, i] > 0)[0]
        j_indices = np.where(np.all(M['li'][ki, :] == M['li'][ki, i, None], axis=0))[0]
        for j in j_indices:
            if j != i and np.count_nonzero(ki) < np.count_nonzero(M['li'][:, j] > 0):
                r.append(j)

    # Remove duplicates from r
    r = list(set(r))

    # Removing elements
    M['li'] = np.delete(M['li'], r, axis=1)
    M['C'] = np.delete(M['C'], r)
    M['X'] = np.delete(M['X'], r, axis=1)
    M['Y'] = np.delete(M['Y'], r, axis=1)

    print(f'{len(r)} removed')
    return M

# Example usage
# M = {'li': np.array(...), 'C': np.array(...), 'X': np.array(...), 'Y': np.array(...)}
# M = remove_overlaps(M)

def normx(x):
    """
    NNormalize each column of a matrix so that the norm of each column is 1.

    Args:
    x: A numpy array of size (M, N).

    Returns:
    A numpy array where each column has been normalized.

    Notes:
    The function normx takes a NumPy array x as input.
    It first checks if the array is not empty (x.size != 0).
    It calculates the L2 norm (Euclidean norm) for each column using np.linalg.norm.
    Zero norms are replaced with 1 to avoid division by zero.
    Each column of x is then normalized by dividing by its norm.
    The function returns the normalized matrix.
    """
    if x.size != 0:
        # Compute the Euclidean norm (L2 norm) for each column.
        # np.linalg.norm computes the norm along the first axis (i.e., along each column)
        # Keepdims=True makes the output shape (M, 1) instead of (M,) for correct broadcasting
        norms = np.linalg.norm(x, axis=0, keepdims=True)

        # Avoid division by zero by replacing zero norms with 1
        norms[norms == 0] = 1

        # Normalize each column by its norm
        x = x / norms

    return x

# Example usage
# x = np.array([[1, 2, 3], [4, 5, 6]])
# normalized_x = normx(x)
# print("Normalized matrix:\n", normalized_x)

def norml(l):
    """
    Normalize each hyperplane in matrix l.

    Multiplies each hyperplane by a scalar so that, for each column n,
    the norm of l[:-1, n] (all but the last row) is 1.

    Args:
    l: A numpy array representing a set of hyperplanes.

    Returns:
    A numpy array with normalized hyperplanes.

    Notes:
    It calculates the L2 norm (Euclidean norm) for each column of l, excluding the last row.
    To avoid division by zero, it replaces any zero norms with 1.
    It normalizes each hyperplane by dividing by its norm.
    Note the use of norms[:, np.newaxis] to ensure proper broadcasting during division.
    This function assumes that l is a NumPy array. 
    """

    # Compute the L2 norm of each hyperplane excluding the last component
    norms = np.sqrt(np.sum(l[:, :-1] ** 2, axis=1))

    # Assuming l is a numpy array with exactly 3 columns
    # Only use the first two elements for norm calculation
    #norms = np.sqrt(np.sum(l[:, :2] ** 2, axis=1))  

    # Avoid division by zero
    norms[norms == 0] = 1

    # Normalize each hyperplane
    l_normalized = l / norms[:, np.newaxis]

    return l_normalized

# Example usage
#l = np.random.rand(4, 3)  # Random 4x3 matrix representing hyperplanes
#l_normalized = norml(l)
#print("Normalized hyperplanes:\n", l_normalized)

def hom(x):
    # Convert input to a 2D numpy array if it is not already
    x = np.atleast_2d(x)
    
    # Add a row of ones for homogeneous coordinates
    ones_row = np.ones((x.shape[0], 1))
    return np.hstack([x, ones_row])

def nhom(x):
    """
    Convert points from homogeneous to non-homogeneous coordinates.

    Args:
    x: A numpy array where each column represents a point in homogeneous coordinates.

    Returns:
    A numpy array where each column represents the same point in non-homogeneous coordinates.
    """
    if x.size == 0:
        return np.array([])  # Return an empty array if x is empty

    d = x.shape[0] - 1  # The dimension of the non-homogeneous coordinates
    # Divide each element by the last row of the array
    x = x[:d, :] / x[-1, :]

    return x

# Example usage
#x_hom = np.array([[2, 4, 6], [4, 8, 12], [2, 2, 2]])  # Homogeneous coordinates
#x_cart = nhom(x_hom)  # Convert to Cartesian coordinates
#print("Non-homogeneous (Cartesian) coordinates:\n", x_cart)

def boxmeet(a, b):
    """
    Compute the intersection of two boxes.

    Args:
    a, b: Numpy arrays of shape (N, 2), where each row represents a box defined by its sorted corner coordinates.
          The first column should contain the minimum coordinate (lower left corner),
          and the second column should contain the maximum coordinate (upper right corner).

    Returns:
    c: A numpy array of shape (N, 2) representing the intersection of the boxes.
       If the boxes are disjoint, returns an array with NaN values.
    """
    # Compute the intersection of the boxes
    c = np.zeros_like(a)
    c[:, 0] = np.maximum(a[:, 0], b[:, 0]) #taking the maximum of the lower bounds
    c[:, 1] = np.minimum(a[:, 1], b[:, 1]) #taking the minimum of the upper bounds

    # Check if any box is disjoint or if any box has NaN values
    if np.any(c[:, 0] > c[:, 1]) or np.any(np.isnan(a)) or np.any(np.isnan(b)):
        c = np.ones_like(a) * np.nan  # Fill with NaN if disjoint or input contains NaN

    return c #returns the intersection of the boxes, with NaN values for disjoint boxes or if the input contains NaN.

# Example usage
#a = np.array([[1, 3], [4, 6]])  # Two boxes in 'a'
#b = np.array([[2, 4], [5, 7]])  # Corresponding boxes in 'b'
#c = boxmeet(a, b)  # Intersection of boxes
#print("Intersection of boxes:\n", c)

def linesegs2H_oriproj(u1, u2, P1, P2):
    """
    Compute a homography H consistent with two camera projection matrices P1 and P2,
    transforming a line segment in the first image to the corresponding segment in the second image.

    Args:
    u1: Endpoints of the line segment in the first image.
    u2: Endpoints of the corresponding line segment in the second image.
    P1, P2: The camera projection matrices.

    Returns:
    H: The computed homography matrix.
    """
    # Compute the fundamental matrix and epipole
    F = vgg.F_from_P(P1, P2)
    e2 = P2 @ vgg.wedge(P1)

    # Normalize F and e2
    F = normx(F.ravel()).reshape(F.shape)
    e2 = normx(e2)

    # Compute initial homography H0
    H0 = vgg.contreps(e2) @ F

    # Compute parameters for the 1-parameter family of solutions
    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    s = np.zeros(2)
    for n in range(2):
        a[:, n] = vgg.contreps(np.hstack([u2[:, n], 1])) @ H0 @ np.hstack([u1[:, n], 1])
        b[:, n] = vgg.contreps(np.hstack([u2[:, n], 1])) @ e2
        s[n] = -np.linalg.solve(b[:, n], a[:, n])

    m0 = s / hom(u1)
    l1 = vgg.wedge(np.vstack([u1, np.ones((1, 2))]))

    # Update H0 based on computed parameters
    H0 += e2 @ m0

    # Compute Jacobian determinant for middle of the line segment
    M = (np.eye(2) - u2[:, [0]]) / (H0[2, :] @ np.hstack([u1[:, 0], 1]))
    M0 = M @ H0[:, :2]
    Mt = M @ e2 @ l1[:2]

    a = np.linalg.det(np.hstack([M0[:, [0]], Mt[:, [1]]])) + np.linalg.det(np.hstack([Mt[:, [0]], M0[:, [1]]]))
    t = (1 - np.linalg.det(M0)) / a

    # Final homography matrix
    H = H0 + t * e2 @ l1

    return H

# Example usage
# Note: This example assumes you have defined the matrices u1, u2, P1, and P2 appropriately.
# u1 = np.array([[x1, y1], [x2, y2]])
# u2 = np.array([[x1_prime, y1_prime], [x2_prime, y2_prime]])
# P1 = np.array([...])  # Camera projection matrix for the first image
# P2 = np.array([...])  # Camera projection matrix for the second image
# H = linesegs2H_oriproj(u1, u2, P1, P2)
# print("Computed homography H:\n", H)

def linesegs2H_metric(P1, P2, L):
    """
    Compute homography H mapping neighborhood of segment in image 1
    to neighborhood of segment in image 2, consistent with camera matrices.

    Args:
    P1, P2: Camera projection matrices.
    L: Scene line in Plücker coordinates.

    Returns:
    H: Computed homography matrix.
    B: Matrix used in the computation of the homography.
    """
    # Compute matrix L by contracting with the epsilon tensor
    L = vgg.contreps(np.outer(L[:, 0], L[:, 1]) - np.outer(L[:, 1], L[:, 0]))

    # Compute scene point at infinity (normal of plane l1*P1)
    X = hom(nhom(vgg.wedge(P1)) + nhom(vgg.wedge(P2)))

    # Compute matrices A and B used for homography calculation
    A = X.T @ L
    B = np.hstack([A[:3], np.array([0])]) @ L

    # Compute the homography H induced by the scene plane and the point at infinity
    H = P2 @ vgg.H_from_P_plane(B, P1)

    return H, B

# Example usage
# P1, P2 are the camera projection matrices
# L is the scene line in Plücker coordinates
# P1 = np.array([...])
# P2 = np.array([...])
# L = np.array([...])
# H, B = linesegs2H_metric(P1, P2, L)
# print("Computed homography H:\n", H)

def remove_beyond_infty(M, P):
    print('Removing matches beyond infinity .. ', end='')
    r = []

    for i in range(M['li'].shape[1]):
        r.append(False)
        for k in np.where(M['li'][:, i] > 0)[0]:
            if np.any(np.dot(np.array([0, 0, 1]), np.dot(P[k], hom(nhom(np.vstack([M['X'][:, i], M['Y'][:, i]])))) < 0)):
                r[-1] = True
                break

    r = np.array(r)
    M['li'] = np.delete(M['li'], r, axis=1)
    M['C'] = np.delete(M['C'], r)
    M['X'] = np.delete(M['X'], r, axis=1)
    M['Y'] = np.delete(M['Y'], r, axis=1)

    print(f'{np.count_nonzero(r)} removed')
    return M

# Example usage
# M = {'li': np.array(...), 'C': np.array(...), 'X': np.array(...), 'Y': np.array(...)}
# P = [np.array(...), ...]  # List of projection matrices
# M = remove_beyond_infty(M, P)

def lmatch_photoscore(u1, u2, P1, P2, I1, I2, CorrParams, metric, L=None):
    """
    Compute similarity score of two line segments for individual pixels.

    Args:
    u1, u2: Line segments in image 1 and image 2 (2x2 arrays).
    P1, P2: Camera matrices.
    I1, I2: Images (2D arrays).
    CorrParams: Correlation window parameters [wr_orthog, wr_parall, pixdist, both_sides].
    metric: Indicates whether metric calibration is assumed (1) or projective (0).
    L: 3D infinite line in Plücker coordinates (optional).

    Returns:
    c: Similarity score vector for the line points.
    """
    #check if the 3D line L is given. If not, it reconstructs L from the image lines.
    if L is None:
        # Reconstruct 3D line L from image lines
        l1 = cros(hom(u1)).T
        l2 = cros(hom(u2)).T
        L = np.linalg.null_space(np.vstack([l1 @ P1, l2 @ P2]))
    else:
        # Project L to images
        l1 = cros(P1 @ L).T
        l2 = cros(P2 @ L).T

    # Segment in image 2 projected onto L
    n = np.array([-l2[1], l2[0]])
    X2 = np.linalg.solve(np.vstack([n, -n @ u2[:, 0], n, -n @ u2[:, 1]]), P2 @ (np.outer(L[:, 0], L[:, 1]) - np.outer(L[:, 1], L[:, 0]))).T

    # Clip segment in image 1 by reprojected segment from image 2
    Q = subtx(hom(u1))
    iQ = subinv(Q)
    t1 = np.sort(nhom(iQ @ hom(u1)), axis=None)
    t2 = np.sort(nhom(iQ @ P1 @ X2), axis=None)
    t = boxmeet(np.array([t1]).T, np.array([t2]).T)

    if np.any(np.isnan(t)):
        return np.array([0])

    v1 = nhom(Q @ hom(t))
    if np.dot(np.array([1, -1]), u1.T @ v1 @ np.array([1, -1])) > 0:
        u1 = v1
    else:
        u1 = v1[:, [1, 0]]

    # Compute homography mapping of neighborhoods of the line segments
    # Depending on whether metric calibration is assumed
    if metric:
        H, _ = linesegs2H_metric(P1, P2, L)
    else:
        H = linesegs2H_oriproj(nhom(P1 @ X2), u2, P1, P2)

    # Compute similarity score
    c = scoreH(u1, H, I1, I2, CorrParams)

    return c

def lineseg3d_from_L(s, P, L):
    """
    Reconstructs a single 3D line segment from its image projections.

    The function projects the end points of the segment onto reprojected straight lines, then 
    reprojects all points into the first image and determines the most distant points to reconstruct 
    the 3D segment.

    Args:
    s: List of line segments in each image.
    P: List of camera projection matrices.
    L: 3D line in Plücker coordinates.

    Returns:
    X: Homogeneous end points of the 3D segment.
    """
    K = len(P)

    # Project segment's end points on reprojected straight lines
    Pj = np.vstack(P)
    l = vgg_wedge(np.reshape(Pj @ L[:, 0], [3, K]), np.reshape(Pj @ L[:, 1], [3, K]))
    x = []
    for k in range(K):
        Q = subtx(l[k, :])
        x.append(Q @ subinv(Q) @ hom(np.vstack([s[k].u, s[k].v])))

    # Reproject all end points into the first image
    x1 = x[0]
    for k in range(1, K):
        x1 = np.hstack([x1, vgg_contreps(l[0, :]) @ vgg_F_from_P([P[k], P[0]]) @ x[k]])

    # Find the most distant points of x1
    t = nhom(subinv(Q[0]) @ x1)
    t = np.array([np.min(t), np.max(t)])
    t0 = nhom(subinv(Q[0]) @ hom(np.vstack([s[0].u, s[0].v])))
    if np.prod(np.array([t, t0]) @ np.array([1, -1])) < 0:
        t = t[::-1]  # Preserve segment orientation
    x = Q[0] @ hom(t)

    # Reconstruct 3D end points
    A = l[1, :] @ P[1]
    X = normx(np.vstack([
        vgg_contreps(P[0].T @ vgg_contreps(x[:, 0]) @ P[0]) @ A,
        vgg_contreps(P[0].T @ vgg_contreps(x[:, 1]) @ P[0]) @ A
    ]).T)

    return X

# Example usage
# Note: This example assumes you have defined the matrices u1, u2, P1, P2, I1, I2, and CorrParams appropriately.
# metric = 1 or 0 based on calibration
# L is optional and may be defined based on your data
# c = lmatch_photoscore(u1, u2, P1, P2, I1, I2, CorrParams, metric, L)
# print("Similarity score:", c)

def basepair(M, kk, l, P, I, opt):
    if opt['Calibration'] > 1:
        A = np.array([0, 0, 0, 1])  # Plane at infinity
    else:
        A = np.array([])

    u2 = hom(np.array([l[kk[1]]['u']]))
    v2 = hom(np.array([l[kk[1]]['v']]))
    EB = epibeam_init(u2, v2, P[kk[0]], A)
    ll2 = norml(np.vstack([l[kk[1]]['l']]))

    Mli = M['li'][kk]
    li = np.array([], dtype=np.int32)
    C, X, Y = [], [], []

    # Assuming vgg_hash is a custom function in your MATLAB code; you'll need to implement it in Python
    vgg_hash = vgg.VGGHash()
    vgg_hash([1, len(l[kk[0]])])
    for n1 in range(1, len(l[kk[0]]) + 1):

        i1 = (Mli[0] == n1)
        l1 = l[kk[0]][n1 - 1]
        uv1 = hom(np.array([[l1['u'], l1['v']]]))

        nn2 = epibeam_search(EB, uv1)
        # Implement the logic of 'all' and 'abs' as per MATLAB's behavior
        nn2 = nn2[all(np.abs((ll2[nn2] @ uv1.T).T) <= opt['DispRange'])]

        for n2 in nn2:
            if any(i1 & (Mli[1] == n2)):
                continue

            l2 = l[kk[1]][n2]
            c = photoscore([l1['u'], l1['v']], [l2['u'], l2['v']], P[kk], I[kk], opt['NCCWindow'], opt['Calibration'] >= 3)
            c = c[c > opt['NCCThreshold'][0]]
            if len(c) < opt['NCCThreshold'][1] / opt['NCCWindow'][2]:
                c = 0
            else:
                c = np.mean(c)

            if c > 0:
                # Reconstruct 3D line segment
                ln = np.array([l[kk[0]][n1 - 1], l[kk[1]][n2]])
                Ln = vgg.line3d_from_lP_lin([vgg.vech(ln[0]['s']), vgg.vech(ln[1]['s'])], [P[k] for k in kk], [I[k].shape for k in kk])
                Ln = lineseg3d_from_L(ln, [P[k] for k in kk], Ln)

                # Implement the condition check for plane at infinity
                if opt['Calibration'] > 1 and any_condition_here:
                    continue

                li = np.append(li, [[n1], [n2]], axis=1)
                C.append(c)
                X.append(Ln[:, 0])
                Y.append(Ln[:, 1])

        vgg_hash(n1)

    # Update M
    if li.size > 0:
        li[kk] = li
        other_indices = [i for i in range(M['li'].shape[0]) if i not in kk]
        li[other_indices] = -1
        M['li'] = np.hstack([M['li'], li.astype(np.int32)])
        M['C'] = np.hstack([M['C'], -np.log(1 - np.array(C))])
        M['X'] = np.hstack([M['X'], X])
        M['Y'] = np.hstack([M['Y'], Y])

    return M

# Define the helper functions like epibeam_search, lmatch_photoscore, etc.

def epibeam_init(x2, y2, P1, P2, A=None):
    """
    Initializes epibeam_search with necessary parameters.

    Args:
    x2, y2: Arrays of shape (2, N2), representing endpoints of line segments in image 2.
    P1, P2: Camera matrices, arrays of shape (3, 4).
    A: Clipping planes, typically the plane at infinity, optional.

    Returns:
    EB: A dictionary containing the initialized parameters for epibeam search.

    Notes:
    The function epibeam_init takes the endpoints of line segments (x2, y2), camera matrices (P1, P2), and optionally clipping planes (A).
    It computes the fundamental matrix F from the camera matrices, normalizes it, and stores it in the EB dictionary.
    It calculates and normalizes epipoles e1 and e2.
    If clipping planes A are provided, it computes the matrix H.
    It determines which side of the epipole each line segment lies on and stores this information in EB.
    The function assumes the presence of vgg_F_from_P, vgg_wedge, and vgg_H_from_P_plane functions in Python.
    """

    # Initialize the epibeam structure
    EB = {}

    # Compute the fundamental matrix from camera matrices
    EB['F'] = vgg.F_from_P([P1, P2])
    # Normalize F
    EB['F'] /= np.linalg.norm(EB['F'])

    # Compute and normalize e1 and e2
    EB['e1'] = vgg.wedge(P2)
    EB['e1'] = P1 @ EB['e1']
    EB['e1'] /= np.linalg.norm(EB['e1'])

    EB['e2'] = vgg.wedge(P1)
    EB['e2'] = P2 @ EB['e2']
    EB['e2'] /= np.linalg.norm(EB['e2'])

    # Store x2 and y2
    EB['x2'] = x2
    EB['y2'] = y2

    # Initialize H as an empty matrix
    EB['H'] = np.zeros((3, 0))

    # If clipping planes A are provided, compute H
    if A is not None:
        for n in range(A.shape[0]):
            H_P_plane = vgg.H_from_P_plane(A[n, :], P2)
            EB['H'] = np.hstack([EB['H'], P1 @ H_P_plane])

    # Determine the side of the epipole for each line segment
    EB['p2'] = np.array([np.linalg.det(np.vstack([x2[:, n], y2[:, n], EB['e2']])) > 0 for n in range(x2.shape[1])])

    return EB

# Example usage
#x2 = np.random.rand(2, 5)
#y2 = np.random.rand(2, 5)
#P1 = np.random.rand(3, 4)
#P2 = np.random.rand(3, 4)
#A = np.random.rand(4, 1)  # Optional
#EB = epibeam_init(x2, y2, P1, P2, A)
#print("Initialized EB:", EB)

def epibeam_search(EB, x1):
    """
    Finds segments in image 2 satisfying the epipolar beam constraint with respect to a given segment in image 1.

    Args:
    EB: Output of epibeam_init.
    x1: A numpy array of shape (3, 2), representing a query segment in image 1 (in homogeneous coordinates).

    Returns:
    n2: Indices of segments in image 2 that satisfy the constraint.

    Notes:
    The function epibeam_search takes the EB structure (output from epibeam_init) and a segment x1 from the first image.
    It first checks the orientation of x1 relative to the epipole e1.
    It then filters out line segments in the second image based on their orientation and position relative to the epipolars of x1.
    The function uses NumPy operations like matrix multiplication (@), determinant calculation (np.linalg.det), and array slicing for efficient computation.
    Note that np.cross is used to compute the cross product.
    """

    # Determine the orientation of the line x1 relative to the epipole e1
    p1 = np.linalg.det(np.hstack([x1, EB['e1'].reshape(-1, 1)])) > 0

    # Find lines in image 2 that have consistent orientation with line x1
    n2 = np.where(EB['p2'] != p1)[0]
    if n2.size == 0:
        return n2

    # Compute the epipolars of line x1 in image 2
    m2 = (EB['F'] @ x1).T

    # Filter segments based on their position relative to the epipolars
    sign = -1 if p1 else 1
    n2 = n2[sign * m2[0, :] @ EB['y2'][:, n2] < 0]
    if n2.size == 0:
        return n2

    n2 = n2[sign * m2[1, :] @ EB['x2'][:, n2] > 0]
    if n2.size == 0:
        return n2

    # Remove segments that are behind any clipping plane
    l2 = np.dot((sign * np.cross(x1.T)).T, EB['H']).reshape((3, -1)).T
    n2 = n2[np.all(l2 @ EB['x2'][:, n2] >= 0, axis=0)]
    n2 = n2[np.all(l2 @ EB['y2'][:, n2] >= 0, axis=0)]

    return n2

# Example usage
# Assume EB is the output from epibeam_init
# x1 is a query segment in image 1 (in homogeneous coordinates)
# x1 = np.array([...])
# n2 = epibeam_search(EB, x1)
# print("Segments in image 2 satisfying the constraint:", n2)


def uv2pixels(uv, step):
    """
    Given endpoints of a line segment, returns pixels from u to v with a specified step.

    Args:
    uv: A numpy array of shape (2, 2), representing the endpoints of a line segment.
    step: The step size for generating pixel coordinates along the line segment.

    Returns:
    x: A numpy array containing the pixel coordinates along the line segment.

    Notes:
    The function uv2pixels takes a 2x2 NumPy array uv (representing endpoints u and v) and a step size.
    It computes the difference du between u and v.
    It calculates the unit direction vector p and the scalar projection t.
    It generates points along the line segment using a NumPy array for t_values and broadcasts these values to compute the coordinates x.
    """

    # Compute the difference between the endpoints
    du = uv @ np.array([-1, 1])

    # Calculate the unit direction vector
    p = du / np.sqrt(np.dot(du, du))

    # Compute the scalar projection of du onto p
    t = np.dot(p, du)

    # Generate points along the line segment
    t_values = np.arange(0, t + step * np.sign(t), step * np.sign(t))
    x = p[:, np.newaxis] @ t_values[np.newaxis, :] + uv[:, 0, np.newaxis]

    return x

# Example usage
#uv = np.array([[1, 4],  # Endpoint u
#               [3, 6]]) # Endpoint v
#step = 0.5
#pixels = uv2pixels(uv, step)
#print("Pixels along the line segment:", pixels)

import numpy as np

def scoreH(u1, H, I1, I2, CorrParams):
    """
    Compute a correlation score for each pixel of a line segment.

    Args:
    u1: Endpoints of the line segment in the first image.
    H: Homography matrix relating the first image to the second.
    I1, I2: The two images.
    CorrParams: Parameters for the correlation computation.

    Returns:
    A list of correlation scores for each pixel on the line segment.
    """
    # Extract correlation window parameters
    wr = CorrParams[:2]  # Half-sides of the correlation window
    pixdist = CorrParams[2]  # Distance between neighboring correlation windows on the line
    mode = CorrParams[3]  # Correlation mode (0 for middle, 1 for all parts)

    # Compute displacements for the correlation window
    i = np.outer(np.ones(2 * wr[1] + 1), np.arange(-wr[0], wr[0] + 1))
    j = np.outer(np.arange(-wr[1], wr[1] + 1), np.ones(2 * wr[0] + 1))

    # Define indices for different parts of the line segment
    kk = [np.arange(1, 2 * wr[0] + 2),
          np.arange(1, wr[0] + 2),
          np.arange(wr[0] + 1, 2 * wr[0] + 2)]

    # Convert line segment endpoints to pixels
    p1 = uv2pixels(u1, pixdist)

    # Rotate to line direction
    l1 = np.cross(hom(u1)).reshape(-1, 1)
    d1 = normx(l1[:2])
    R1 = np.hstack([d1, np.array([[0, -1], [1, 0]]) @ d1])

    # Initialize correlation scores
    c = np.full(p1.shape[1], -1)

    # Determine which parts of the line to use for correlation
    K = [0] if mode == 0 else [0, 1, 2]

    # Loop for correlating different parts of the line segment
    for k in K:
        ki = i[:, kk[k]]
        kj = j[:, kk[k]]
        w = R1 @ np.vstack([ki.ravel(), kj.ravel()])
        w = np.vstack([w, np.ones(w.shape[1])])
        c = np.maximum(c, vgg_ncc_2im_H(I1, I2, H, w, p1))

    return c

def subtx(S):
    """
    Compute orthonormal parameterization of a subspace of a linear space.
    Depending on the shape of S, it either directly uses S or computes a unitary matrix P that spans the subspace.

    Args:
    S: A matrix representing the subspace. If S is D x K, it's defined by join of K points.
       If S is K x D, it's defined by meet of K hyperplanes.

    Returns:
    P: An orthogonal matrix with columns spanning the subspace.
    x0: The affine origin, orthogonal projection of origin to S.

    Notes:
    The function applies a Householder transformation to make the last row of P (except the last element) zeros.
    Adjusts the sign of P based on the determinant to ensure det(R) > 0 where subtx(S) @ R = S.
    Normalizes the last column of P to make its last element 1.
    Splits P into an affine projection matrix p and an affine origin x0.
    """
    N, K = S.shape

    # P: unitary matrix with columns spanning the subspace
    if N >= K:
        if np.any(np.abs(np.eye(K) - S.T @ S) > 100 * np.finfo(float).eps):
            # S is not orthogonal, apply Singular Value Decomposition
            P, s, U = scipy.linalg.svd(S, full_matrices=False)
        else:
            P = S
    else:
        P = scipy.linalg.null_space(S)
        N, K = P.shape

    # Make P(end, 1:end-1) = 0 using a householder transformation
    P = P @ householder(normx(P[-1, :]).T, np.append(np.zeros((K - 1, 1)), 1))

    # Adjust sign of P
    if S.shape[0] > S.shape[1]:
        P = P * np.sign(np.linalg.det(subinv(P) @ S) * P[-1, -1])

    # Normalize P(end, end) to 1
    P[:, -1] = P[:, -1] / P[-1, -1]

    x0 = None
    if P.shape[1] > 1:
        x0 = P[:-1, -1]
        P = P[:-1, :-1]

    return P, x0

def householder(x, y):
    """
    Compute a Householder reflection matrix that transforms vector x to y.
    Both x and y should be unit vectors.

    Args:
    x: A unit vector.
    y: Another unit vector.

    Returns:
    R: The Householder reflection matrix that transforms x to y.
    """
    # Ensure x and y are unit vectors
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    # Choose correct sign of y to reduce round-off error
    # The choice is based on the dot product of x and y.
    # Depending on the sign, it computes the vector u either as x - y or x + y.
    # Finally, it constructs the Householder matrix R. This matrix is orthogonal (R.T == R) and reflects x into y.
    if np.dot(x, y) < 0:
        u = x - y
        R = np.eye(len(u)) - 2 * np.outer(u, u) / np.dot(u, u)
    else:
        u = x + y
        R = -np.eye(len(u)) + 2 * np.outer(u, u) / np.dot(u, u)

    return R

# Example usage
#x = np.array([1, 0, 0])
#y = np.array([0, 1, 0])
#R = householder(x, y)
#print("Householder matrix R:\n", R)

def subinv(P):
    """
    Compute the inverse of a transformation matrix obtained from subtx.

    Args:
    P: A transformation matrix from subtx.

    Returns:
    The inverse transformation matrix.
    """
    # Extract the upper left (D-1) x (D-1) submatrix and transpose it
    p = P[:-1, :-1].T

    # Compute the transformation for the inverse
    # The last row of the original matrix P is used in the calculation
    P_inv = np.vstack([p - p @ P[:-1, -1] / P[-1, -1], np.zeros((1, P.shape[0] - 1))])
    P_inv[-1, -1] = 1  # Set the last element to 1

    return P_inv  #returns the inverse transformation matrix

# Example usage
# P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Example transformation matrix
# P_inv = subinv(P)
# print("Inverse transformation matrix:\n", P_inv)


def generate(M, l, P, I, D, kk, opt):
    # Initialize list M of tentative matches if it's empty
    if M is None:
        M = {
            'li': np.zeros((len(P), 0), dtype=np.int32),
            'C': [],
            'X': [],
            'Y': []
        }

    # Print out initial state and options
    print(f'GENERATING TENTATIVE MATCHES BASED ON VIEW PAIR [{kk[0]} {kk[1]}]')
    for k in range(len(P)):
        print(f'{len(l[k])} ', end='')
    print(f'Number of views: {len(P)}')
    print('Numbers of line segments: ', end='')
    for k in range(len(P)):
        print(f'{len(l[k])} ', end='')
    print()
    print(f'Displacement search range [pixels]: {opt["DispRange"]}')
    print(f'Maximal view distance [views]: {opt["MaxViewDistance"]}')
    calibration_options = ['projective', 'affine', 'metric']
    if opt['Calibration'] < 1 or opt['Calibration'] > 3:
        raise ValueError('Wrong option: Calibration')
    calibration = calibration_options[opt['Calibration'] - 1]
    print(f'Calibration: {calibration}')
    print(f'Max. reproj. residuals [pixels_lin pixels_nonlin]: [{opt["ReprojResidual"][0]} {opt["ReprojResidual"][1]}]')
    #print(f'Correl. window half size [pixels_perp pixels_paral]: [{opt["CorrParams"][0]} {opt["CorrParams"][1]}]')
    #print(f'Correl. window distance [pixels]: {opt["CorrParams"][2]}')
    #window_placement_options = ['center', 'center_and_sides']
    #window_placement = window_placement_options[opt['CorrParams'][3]]
    #print(f'Correl. window placement: {window_placement}')

    # Precompute straight lines and covariance matrices for line segments
    for k in range(len(P)):
        for n in range(len(l[k])):
            u, v = l[k][n]['u'], l[k][n]['v']
            u_hom = hom(u) #np.append(u, 1)
            v_hom = hom(v) #np.append(v, 1)
            cross = np.cross(u_hom, v_hom)
            l[k][n]['l'] = norml(cross)
            #l[k][n]['l'] = cross / np.linalg.norm(cross)
            if 's' not in l[k][n]:
                uv = np.column_stack((u, v))
                #print(uv)
                uv_hom = hom(uv) #np.hstack((uv, np.ones((uv.shape[0], 1))))
                #print(uv_hom)
                l[k][n]['s'] = vgg.vech(np.dot(uv_hom, uv_hom.T))  # Replace 'vgg_vech' with Python equivalent

    # Call the print_stat function with M.li as the input
    print(M)
    print("Histogram of matches [#views_per_match:#matches]:", end=' ')
    print_stat(M['li'])

    # Check if the distance of the base pair is greater than the maximum view distance
    if D[kk[0], kk[1]] > opt['MaxViewDistance']:
        raise ValueError("Distance of base pair is greater than maximal view distance")

    M = basepair(M, kk, l, P, I, opt)
    M = remove_overlaps(M)
    if opt['Calibration'] > 1:
       M = remove_beyond_infty(M, P)
    print_stat(M['li'])

    return M

# Additional helper functions like 'normalize', 'homogenize', 'vgg_vech' need to be defined

def lmatch_detect_lines(I, minLength):
    """
    Detects line segments in image I with a length greater than minLength.
    
    Parameters:
    I (numpy.ndarray): The input image.
    minLength (int): The minimal length of lines to be detected.
    
    Returns:
    tuple: Two numpy arrays representing the start (u) and end (v) points of the detected line segments.
    """
    # Edge detection using Canny
    edges = cv2.Canny(I, threshold1=20, threshold2=100, apertureSize=3, L2gradient=True)
    
    # Find contours (edgel strips) from Canny edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Convert contours to the expected format for vgg_linesegs_from_edgestrips
    e = [contour[:, 0, ::-1].T for contour in contours]  # Reverse X and Y to match MATLAB format
    
    # Detect line segments from edgel strips
    u, v, _ = vgg.vgg_linesegs_from_edgestrips(e)  # Assuming implementation is provided
    
    # Filter lines based on length
    lengths = np.sum((u - v) ** 2, axis=0)
    indices = np.where(lengths >= minLength**2)[0]
    u = u[:, indices]
    v = v[:, indices]
    
    return u, v