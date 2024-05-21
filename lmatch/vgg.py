import sys
import numpy as np
from scipy.special import gammainc

def vgg_bilinear_interpolation(image, i, j, x_invalid=None):
    """
    Perform bilinear interpolation for a point (i, j) in the given image.
    If the point is outside the image boundary, return x_invalid if provided, otherwise return 0.

    Args:
    - image: The image array.
    - i, j: The floating point x and y coordinates in the image.
    - x_invalid (optional): Value to return for points outside the image. Defaults to None, which means return 0.

    Returns:
    - Interpolated intensity value at (i, j), or x_invalid/0 for points outside the image.
    """
    ii, ji = int(i), int(j)
    a = i - ii
    b = j - ji

    if 0 <= ii < image.shape[0] - 1 and 0 <= ji < image.shape[1] - 1:
        return (1 - a) * (1 - b) * image[ii, ji] + \
               a * (1 - b) * image[ii + 1, ji] + \
               (1 - a) * b * image[ii, ji + 1] + \
               a * b * image[ii + 1, ji + 1]
    else:
        return 0 if x_invalid is None else x_invalid

def vgg_imgraduv(image, h, v1, v2):
    """
    Image gradient along a line segment.
    """
    s = np.array([v2[0] - v1[0], v2[1] - v1[1]])
    len_s = np.sqrt(np.sum(s**2))
    s /= len_s

    g = 0
    rh = len(h) // 2
    for i in range(-rh, rh + 1):
        hi = h[i + rh]
        sumi = 0
        nsumi = 0
        for j in range(int(len_s)):
            v = np.array([-s[1] * i + s[0] * j + v1[0], s[0] * i + s[1] * j + v1[1]])
            f = vgg_bilinear_interpolation(image, v[0], v[1], -1234567)
            if f != -1234567:
                sumi += hi * f
                nsumi += 1
        g += sumi / nsumi if nsumi else 0
    return g

def vgg_gauss_mask(a, der, range):
    if len(range) != 2 or not all(isinstance(i, int) for i in range):
        raise ValueError("Range must be a tuple of two integers")

    X = np.arange(range[0], range[1] + 1) # +1 because the end value is inclusive
    aa = -0.5 / (a * a)

    def compute_gauss(aa, der, X):
        if der > 0:
            return 2 * aa * ((der - 1) * vgg_compute_gauss(aa, der - 2, X) + X * vgg_compute_gauss(aa, der - 1, X))
        elif der == 0:
            return np.ones_like(X)
        else:
            return np.zeros_like(X)

    return np.exp(aa * X ** 2) / (a * np.sqrt(2 * np.pi)) * vgg_compute_gauss(aa, der, X)

#def vech(m):
#    """
#    Vectorizes or de-vectorizes a symmetric matrix.
#    If m is a matrix, returns a column vector of elements on or below the main diagonal.
#    If m is a vector, returns a symmetric matrix such that vgg_vech(matrix) = m.
#    """
#    m = np.asarray(m)
#    if m.ndim == 1:
#        # De-vectorize
#        N = int((np.sqrt(8 * len(m) + 1) - 1) / 2)
#        r, c = np.triu_indices(N)
#        X = np.zeros((N, N))
#        X[r, c] = m
#        X[c, r] = m
#        return X
#    elif m.ndim == 2:
#        # Vectorize
#        r, c = np.tril_indices_from(m)
#        return m[r, c]
#    else:
#        raise ValueError("Input must be a 1D or 2D array.")

# Example usage
# Vectorize a matrix
# matrix = np.array([[1, 2], [2, 3]])
# vector = vgg_vech(matrix)
# De-vectorize a vector
# vector = np.array([1, 2, 3])
# matrix = vgg_vech(vector)

def vgg_vech(m):
    """
    De-/vectorizes a symmetric matrix.
    
    If input is a square matrix, it returns a column vector of elements on or below
    the main diagonal of the matrix.
    
    If input is a vector (N*(N+1)/2 elements), it returns a symmetric N-by-N matrix
    such that vectorizing this matrix would give the original vector.
    
    Parameters:
    m (numpy.ndarray): A 1D vector or a 2D square matrix.
    
    Returns:
    numpy.ndarray: A vector or a symmetric matrix, depending on the input.
    """
    M, N = m.shape if m.ndim > 1 else (1, len(m))
    
    if M == 1 or N == 1:  # Vector to matrix conversion
        N = int((np.sqrt(8*M*N + 1) - 1) / 2)  # Solve quadratic equation for N
        h = np.zeros((N, N))
        indices = np.tril_indices(N)
        h[indices] = m[:indices[0].size]
        h = h + h.T - np.diag(np.diag(h))  # Make symmetric without doubling diagonal
    else:  # Matrix to vector conversion
        indices = np.tril_indices_from(m)
        h = m[indices]
        
    return h

def vgg_conv_lineseg(image, h, u1, u2):
    """
    Convolve a line segment in an image using a specified kernel h.
    :param image: Input image.
    :param h: Convolution kernel.
    :param u1: Start points of line segments.
    :param u2: End points of line segments.
    :return: Convolution results for each line segment.
    """
    if u1.ndim != 2 or u2.ndim != 2:
        raise ValueError("u1 and u2 must be 2D arrays.")
    if u1.shape[0] != 2 or u2.shape[0] != 2:
        raise ValueError("u1 and u2 must have a shape of (2, num_segments).")

    return np.array([vgg_imgraduv(image, h, u1[:, n], u2[:, n]) for n in range(u1.shape[1])])

# Example usage
# image = cv2.imread('path_to_image', cv2.IMREAD_GRAYSCALE)
# h = your_gaussian_derivative_kernel
# u1, u2 = line_segment_endpoints
# result = conv_lineseg(image, h, u1, u2)

class vgg_VGGHash:
    """
    VGGHash is a class that stores the required state across calls.
    When initialized with a 2-element list/tuple, it sets up the range and the total number of hashes (N) to print.
    When called with a scalar, it prints (or returns) the appropriate number of hashes based on the current loop index.
    __call__ method allows the instance of VGGHash to be used like a function.
    """
    def __init__(self):
        self.X = None
        self.N = None
        self.n = None
        self.h = None

    def __call__(self, x, NN=25, hh='#'):
        if isinstance(x, (list, tuple)) and len(x) == 2:  # Initialization call
            self.X = x
            self.N = NN
            if abs(self.X[0] - self.X[1]) <= 1e-15:  # Avoid division by zero
                self.X[0] = self.X[1] - 1
            self.n = 0
            self.h = hh
        elif isinstance(x, (int, float)):  # Update call
            m = int(self.N * (x - self.X[0]) / (self.X[1] - self.X[0]))
            s = self.h * max(m - self.n, 0)
            self.n = m
            print(s, end='')  # Print the hashes directly
        else:
            raise ValueError('x must be a scalar or a 2-element list/tuple')

# Example usage
# vgg_hash = vgg.VGGHash()
# vgg_hash([1, length_of_ui])  # assuming length_of_ui is the length of the loop
# for uii in range(length_of_ui):
#     # ... loop code ...
#     vgg_hash(uii)

def vgg_wedge(*args):
    """
    Compute the wedge product of N-1 N-vectors.
    
    For a single argument X (a matrix), it computes the wedge product of its columns.
    For multiple vector arguments, it computes the wedge product for each (N-1)-tuple of corresponding columns.
    The special case for N=3 (equivalent to the cross product) is handled separately for efficiency.
    For the general case, the wedge product is computed using determinants.
    NumPy is used for matrix and vector operations.
    The np.linalg.det function is used for computing determinants, and NumPy broadcasting rules apply in vectorized operations.
    """

    # Single matrix argument
    if len(args) == 1:
        X = args[0]
        N, Nm1 = X.shape
        
        # If more columns than rows, work with the transpose
        if Nm1 > N:
            return vgg_wedge(X.T).T

        # Special case for N==3 (cross product)
        if N == 3:
            Y = np.array([
                X[1, 0] * X[2, 1] - X[2, 0] * X[1, 1],
                X[2, 0] * X[0, 1] - X[0, 0] * X[2, 1],
                X[0, 0] * X[1, 1] - X[1, 0] * X[0, 1]
            ])
        else:
            # General case
            Y = np.zeros(N)
            for n in range(N):
                #idx = [i for i in range(N) if i != n]
                idx = list(range(n)) + list(range(n + 1, N))
                Y[n] = (-1)**(n + N) * np.linalg.det(X[idx, :])

    # Multiple vector arguments
    else:
        N = len(args) + 1
        if N == 3: # Special case for N==3 (cross product)
            X1, X2 = args
            #Y = np.array([(X1[1, :] * X2[2, :] - X1[2, :] * X2[1, :]),
            #              (X1[2, :] * X2[0, :] - X1[0, :] * X2[2, :]),
            #              (X1[0, :] * X2[1, :] - X1[1, :] * X2[0, :])]).T
            Y = np.array([
                X1[1] * X2[2] - X1[2] * X2[1],
                X1[2] * X2[0] - X1[0] * X2[2],
                X1[0] * X2[1] - X1[1] * X2[0]
            ])
        else: # General case
            K = len(args[0]) #args[0].shape[1]
            Y = np.zeros((K, N))
            for k in range(K):
                #X = np.array([arg[:, k] for arg in args]).T
                X = np.vstack([arg[:, k] for arg in args])
                Y[k, :] = vgg_wedge(X)

    return Y

# Example usage
# X = np.random.rand(3, 2)
# print("Wedge product (single matrix):", vgg_wedge(X))
# X1 = np.random.rand(3, 5)
# X2 = np.random.rand(3, 5)
# print("Wedge product (multiple vectors):", vgg_wedge(X1, X2))

def vgg_F_from_P(P1, P2=None):
    """
    Compute the fundamental matrix from two camera matrices.
    
    Args:
    P1: The first camera matrix.
    P2: The second camera matrix. If not provided, P1 is expected to be a list or tuple of two matrices.
    
    Returns:
    F: The computed fundamental matrix.
    
    The fundamental matrix F is such that, for any point X in 3D space,
    if x1 = P1 * X and x2 = P2 * X are the projections of X onto the two image planes,
    then x2' * F * x1 = 0.

    It uses slicing to extract the required rows from the camera matrices and np.linalg.det to compute the determinants.
    The fundamental matrix F is calculated by stacking the determinants according to the formula given in the MATLAB code.
    np.vstack is used to stack rows vertically, and np.array constructs the matrix F.
    """

    # Handle the case where only P1 is provided and it is a list or tuple containing P1 and P2
    if P2 is None:
        P1, P2 = P1

    # Extract rows from P1 and P2 for determinant calculations
    X1 = P1[[1, 2], :]
    X2 = P1[[2, 0], :]
    X3 = P1[[0, 1], :]
    Y1 = P2[[1, 2], :]
    Y2 = P2[[2, 0], :]
    Y3 = P2[[0, 1], :]

    # Compute the fundamental matrix F
    F = -np.array([[np.linalg.det(np.vstack([X1, Y1])),
                    np.linalg.det(np.vstack([X2, Y1])),
                    np.linalg.det(np.vstack([X3, Y1]))],
                   [np.linalg.det(np.vstack([X1, Y2])),
                    np.linalg.det(np.vstack([X2, Y2])),
                    np.linalg.det(np.vstack([X3, Y2]))],
                   [np.linalg.det(np.vstack([X1, Y3])),
                    np.linalg.det(np.vstack([X2, Y3])),
                    np.linalg.det(np.vstack([X3, Y3]))]])

    return F

# Example usage
# P1 and P2 are 3x4 camera matrices
# P1 = np.random.rand(3, 4)
# P2 = np.random.rand(3, 4)
# F = vgg_F_from_P(P1, P2)
# print("Fundamental Matrix F:", F)

def vgg_H_from_P_plane(A, P):
    """
    Compute a matrix that maps image points to points on a 3D plane.

    Args:
    A: A numpy array of size (4, 1), representing the scene plane.
    P: A numpy array of size (3, 4), representing the camera matrix.

    Returns:
    H: A numpy array of size (4, 3). For an image point x (size (3, 1)),
       the scene point X lying on the scene plane A is given by X = H * x.

    These wedge products are combined to form the matrix H.
    The matrix H is then divided by the dot product of A and the wedge product of P.
    The np.vstack function is used to stack rows vertically, and np.column_stack is used to form the columns of H.
    Ensure the vgg_wedge function is defined or adapted as needed in your Python environment.
    """

    # Compute the wedges required for the matrix H
    col1 = vgg_wedge(np.vstack([A, P[1, :], P[2, :]]))
    col2 = vgg_wedge(np.vstack([P[0, :], A, P[2, :]]))
    col3 = vgg_wedge(np.vstack([P[0, :], P[1, :], A]))

    # Stack the wedges to form the matrix H
    H = -np.column_stack([col1, col2, col3])

    # Divide by the dot product of A and the wedge of P
    H /= np.dot(A, vgg_wedge(P))

    return H

# Example usage
# Define the plane A and camera matrix P
# A = np.random.rand(4, 1)
# P = np.random.rand(3, 4)
# Compute the matrix H
# H = vgg_H_from_P_plane(A, P)
# print("Matrix H:", H)

def vgg_Htimes(H, u):
    """
    Apply homography matrix H to a point u.
    Args:
    - H: Homography matrix.
    - u: Point in homogeneous coordinates.
    Returns:
    - Transformed point in homogeneous coordinates.
    """
    homogeneous_u = np.array([u[0], u[1], 1])
    x = H @ homogeneous_u
    return x / x[2]

def vgg_dhom(H, u):
    """
    Compute the derivative of a homography matrix at point u.
    Args:
    - H: Homography matrix.
    - u: Point in image.
    Returns:
    - Derivative of the homography at point u.
    """
    y = vgg_Htimes(H, u)
    v = y[:2]
    A = np.array([[H[0, 0] - v[0] * H[2, 0], H[0, 1] - v[0] * H[2, 1]],
                  [H[1, 0] - v[1] * H[2, 0], H[1, 1] - v[1] * H[2, 1]]]) / y[2]
    return A

def vgg_ncc_2im_H(X, Y, H, W, method='bilinear'):
    """
    Compute normalized cross-correlation under a homography.
    Args:
    - X, Y: The two images for correlation.
    - H: Homography matrix.
    - W: Weights/displacements matrix.
    - method: Interpolation method ('bilinear' or 'nearest').
    Returns:
    - NCC value. 
    - The function returns the computed NCC value, which measures the similarity between the two images 
      in the specified region under the given homography.
    """
    EX, EY, DX, DY, XY = 0, 0, 0, 0, 0 #variables to accumulate the sums required for NCC calculation
    sumw = 0 #sum of weights
    nw = 0 #number of valid points considered in the NCC calculation.

    for i in range(W.shape[1]): #looping Over Displacement Vectors
        d = W[:2, i] #displacement vector (in X's coordinate system).
        w = W[2, i] #weight for the current displacement vector.

        u = np.array([d[0] - 1, d[1] - 1]) #point in X after applying displacement d.
        v = vgg_Htimes(H, u)[:2] - 1 #corresponding point in Y found by applying the homography H to u.

        #Ensures that the points u and v are within the valid range of their respective images. 
        #If not, skips to the next iteration.
        if u[0] < 1 or u[1] < 1 or u[0] >= X.shape[0] - 1 or u[1] >= X.shape[1] - 1 or \
           v[0] < 1 or v[1] < 1 or v[0] >= Y.shape[0] - 1 or v[1] >= Y.shape[1] - 1:
            continue

        #Retrieves the intensity values at u in X and at v in Y using the specified interpolation method (bilinear or nearest).
        if method == 'bilinear':
            x = vgg_bilinear_interpolation(X, u[0], u[1])
            y = vgg_bilinear_interpolation(Y, v[0], v[1])
        else:
            x = X[int(u[0]), int(u[1])]
            y = Y[int(v[0]), int(v[1])]

        #Accumulates weighted sums
        EX += x * w
        EY += y * w
        DX += x * x * w
        DY += y * y * w
        XY += x * y * w
        sumw += w
        nw += 1

    #Final NCC Calculation
    if nw > 2: #checks if there are enough valid points
        EX /= sumw #normalizes the accumulated sums by sumw.
        EY /= sumw
        DX = DX / sumw - EX * EX #calculates variances and covariance.
        DY = DY / sumw - EY * EY
        XY = (XY / sumw - EX * EY) / np.sqrt(DX * DY) #compute ncc value
        return XY if DX > 1e-9 and DY > 1e-9 else np.nan
    else:
        return np.nan

# Example usage
# X, Y are the two images
# H is the homography matrix
# W is the weights/displacements matrix
# C = ncc(X, Y, H, W)

def vgg_contreps(X):
    """
    Perform contraction with the epsilon tensor on matrix X. This function has specific behaviors 
    depending on the shape and type of the input matrix, transforming vectors into skew-symmetric 
    matrices and vice versa, among other transformations.

    Args:
    X (numpy.ndarray): Input array that can be a 3-vector, a 2-vector, a 3x3 skew-symmetric matrix,
                       or a 4x4 skew-symmetric matrix.

    Returns:
    Y (numpy.ndarray): The result of contracting X with the epsilon tensor.
    """
    shape = X.shape

    # Vector to skew-symmetric matrix (3D cross product matrix)
    if np.prod(shape) == 3:
        Y = np.array([[0, -X[2], X[1]],
                      [X[2], 0, -X[0]],
                      [-X[1], X[0], 0]])
    # Row 2-vector to skew-symmetric matrix
    elif shape == (1, 2):
        Y = np.dot(np.array([[0, 1], [-1, 0]]), X.T)
        #Y = np.array([[0, 1], [-1, 0]]) @ X.T
    # Column 2-vector to skew-symmetric matrix
    elif shape == (2, 1):
        Y = np.dot(X.T, np.array([[0, 1], [-1, 0]]))
        #Y = X.T @ np.array([[0, 1], [-1, 0]])
    # Skew-symmetric 3-by-3 matrix to vector
    elif shape == (3, 3):
        Y = np.array([X[2, 1], X[0, 2], X[1, 0]])
    # Skew-symmetric 4-by-4 matrix (Pluecker matrix dual)
    elif shape == (4, 4):
        Y = np.array([[0, X[2, 3], X[3, 1], X[1, 2]],
                      [X[3, 2], 0, X[0, 3], X[2, 0]],
                      [X[1, 3], X[3, 0], 0, X[0, 1]],
                      [X[2, 1], X[0, 2], X[1, 0], 0]])
    else:
        raise ValueError('Wrong matrix size.')

    return Y

# Example usage
#X_vector = np.array([1, 2, 3])
#Y_matrix = vgg_contreps(X_vector)
#print("Skew-symmetric matrix from vector:\n", Y_matrix)
#X_matrix = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
#Y_vector = vgg_contreps(X_matrix)
#print("Vector from skew-symmetric matrix:\n", Y_vector)

def vgg_fit_hplane_to_x(s):
    """
    Fit a hyperplane to a set of points in N-dimensional space.

    (minimizing the sum of squared orthogonal distances of the points to the hyperplane.)
    The points are provided in the form of an inverse covariance matrix s, which is essentially the 
    matrix product of the points' homogeneous coordinates.

    Args:
    s: Inverse covariance matrix (N+1 x N+1), where each column represents 
       an (N+1)-vector of homogeneous coordinates of points in N-space.

    Returns:
    A: The fitted hyperplane in homogeneous coordinates.
    e: Eigenvalues of the fit.
    """
    N = s.shape[0] - 1  # Space dimension

    # Compute the centroid of the points
    c = s[:N, -1] / s[-1, -1]

    # Perform Singular Value Decomposition (SVD)
    U, e, V = np.linalg.svd(s[:N, :N] - np.outer(c, s[-1, :N]), full_matrices=False)

    # Extract the hyperplane
    A = U[:, -1] #the last column represents the hyperplane A in homogeneous coordinates.
    A = np.hstack([A, -np.dot(A, c)]) #adjust A to account for the centroid c.
    e = np.diag(e) #

    # Normalize the homogeneous part of A
    A[:N] = A[:N] / np.linalg.norm(A[:N])

    return A, e #return the hyperplane A and the eigenvalues e of the fit

# Example usage
# s is the inverse covariance matrix formed by the points' homogeneous coordinates
# s = np.array([...])
# A, e = vgg_fit_hplane_to_x(s)
# print("Fitted hyperplane:", A)
# print("Eigenvalues:", e)

def vgg_conditioner_from_image(c, r=None):
    """
    Create a conditioning matrix for image points based on image dimensions.

    This matrix is typically used in computer vision to normalize the coordinates of image points, 
    improving the numerical stability of various algorithms involving multiple view geometry.

    The function accepts the image dimensions either as separate width (c) and height (r) arguments 
    or as a single tuple/list argument containing both.

    Args:
    c: Image width or a tuple/list containing both width and height.
    r: Image height. Not needed if c is a tuple/list.

    Returns:
    C: Conditioning matrix.
    invC: Inverse of the conditioning matrix (if requested).
    """
    if r is None:
        # Handle the case where c is a tuple/list of [width, height]
        r = c[1]
        c = c[0]

    # Calculate the factor for normalization based on the average of the image dimensions.
    f = (c + r) / 2.0

    # Create the conditioning matrix:
    # It's designed to scale and translate image coordinates such that they are centered around the  
    # origin and normalized in range, typically within a unit square or circle.
    C = np.array([[1.0/f, 0, -c/(2.0*f)],
                  [0, 1.0/f, -r/(2.0*f)],
                  [0, 0, 1]])

    invC = None
    if nargout > 1:
        # Compute the inverse of the conditioning matrix (if requested)
        invC = np.array([[f, 0, c/2.0],
                         [0, f, r/2.0],
                         [0, 0, 1]])

    return C, invC

# Example usage
# c = image_width, r = image_height
# c, r = 640, 480  # Example image dimensions
# C, invC = vgg_conditioner_from_image(c, r)
# print("Conditioning matrix C:\n", C)
# print("Inverse conditioning matrix invC:\n", invC)


def vgg_normx(x):
    """
    Normalize each column of a matrix such that its Euclidean norm is 1.

    Args:
    x: A numpy array where each column represents a vector.

    Returns:
    The normalized numpy array.
    """
    # Calculate the Euclidean norm for each column
    norms = np.sqrt(np.sum(x * x, axis=0))

    # Avoid division by zero
    norms[norms == 0] = 1

    # Normalize each column by its norm
    x = x / norms

    return x

def vgg_norml(l):
    """
    Normalize each column of a hyperplane matrix so that the norm of the non-homogeneous part is 1.

    Args:
    l: A numpy array where each column represents a hyperplane.

    Returns:
    The normalized numpy array.
    """
    # Calculate the Euclidean norm for the non-homogeneous part of each column
    norms = np.sqrt(np.sum(l[:, :-1] ** 2, axis=1))

    # Avoid division by zero
    norms[norms == 0] = 1

    # Normalize each column by its norm
    l = l / norms[:, np.newaxis]

    return l

def vgg_line3d_from_lP_lin(s, P, imsize=None):
    """
    Linear estimation of 3D line from image lines and camera matrices.

    This function is particularly useful in computer vision and photogrammetry for reconstructing 3D 
    lines from their projections in multiple images.

    Args:
    s: List of inverse covariance matrices of the image line segments.
    P: List of camera matrices.
    imsize: List of image sizes for preconditioning (optional).

    Returns:
    L: 3D straight line represented by two homogeneous points.
    """
    K = len(P)  # Number of images

    # Fit hyperplanes to the image lines
    l = np.array([vgg_fit_hplane_to_x(s[k]) for k in range(K)])

    # Preconditioning the camera matrices and the image lines, if needed
    if imsize is not None and K > 2:
        for k in range(K):
            H, invH = vgg_conditioner_from_image(imsize[:, k])
            P[k] = H @ P[k]
            l[k, :] = l[k, :] @ invH
        l = norml(l)

    # Construct matrix M from the image lines and camera matrices
    M = np.vstack([l[k, :] @ P[k] for k in range(K)])

    # Perform SVD and extract the 3D line
    u, s, v = np.linalg.svd(normx(M.T).T, full_matrices=False)
    L = v[:, -2:] #the last two columns of v give the 3D line L in Plücker coordinates.

    return L

# Example usage
# s is a list of inverse covariance matrices for each image line segment
# P is a list of camera matrices
# imsize is a list of image sizes (optional)
# s = [...]
# P = [...]
# imsize = [...]
# L = vgg_line3d_from_lP_lin(s, P, imsize)
# print("Estimated 3D line L:\n", L)


def vgg_get_homg(x):
    """
    Convert a set of non-homogeneous points into homogeneous points.
    
    Parameters:
    x (numpy.ndarray): A 2D array of shape (2, N) containing non-homogeneous points.
    
    Returns:
    numpy.ndarray: A 2D array of shape (3, N) containing homogeneous points.
    """
    if x.size == 0:
        return x
    
    # Append a row of ones to make the points homogeneous
    return np.vstack([x, np.ones((1, x.shape[1]))])


def vgg_findcontig(x):
    """
    Finds contiguous pieces of 1's in a logical vector x.
    
    Parameters:
    x (numpy.ndarray): A 1D array of logical values (True for 1's and False for 0's).
    
    Returns:
    numpy.ndarray: An array of shape (2, N) containing the start and end indices of the contiguous pieces of 1's.
    """
    # Ensure x is a 1D array
    x = np.ravel(x)
    #print(f"x: {x}")

    # Compute differences between adjacent elements
    x_int = x.astype(int)
    d = np.diff(x_int, prepend=x_int[0], append=x_int[-1])
    #print(f"d: {d}")

    # Identify the start and end of contiguous pieces
    starts = np.where(d == -1)[0] + 1  # Adjust by 1 to get the correct start indices
    ends = np.where(d == 1)[0]

    #print(f"starts {starts}")
    #print(f"ends {ends}")
    
    # Combine the start and end indices
    c = np.vstack((starts, ends))
    
    return c


def vgg_chain2segments(x, W, th):
    """
    Segments chains of edgels for polygonal approximation.

    Parameters:
    x (numpy.ndarray): size (2,N), edgel coordinates
    W (int): scalar, radius of the kernel sliding on the edge chain (recommended: 5)
    th (float): scalar, segmentation threshold (recommended: 0.5)

    Returns:
    numpy.ndarray: indices to segments, size (2,M)
    numpy.ndarray: indices to the rejected segments, same format as the segment indices
    """
    N = x.shape[1]
    if N < 2:
        return np.array([]), np.array([])
    
    px = vgg_get_homg(x)

    t = np.zeros(N, dtype=bool) # int or bool?

    # Sliding segment
    r = np.nan * np.ones(N)
    for i in range(W, N-W):
        l = px[:, i+W].T @ vgg_contreps(px[:, i-W]) 
        d = l / np.sqrt(np.sum(l[0:2] ** 2)) @ px[:, i-W:i+W+1] #FIXME denominator goes to zero sometimes
        r[i] = np.max(np.abs(d))
        if r[i] < th:
            t[i-W+1:i+W] = True

    #print(t)

    ij = vgg_findcontig(t)
    #if ij.size > 0:
    #    ij = ij[:, ij[1, :] - ij[0, :] >= 3]

    ijn = vgg_findcontig(~t)

    return ij, ijn


def vgg_hplanefit(x, sigma=None):
    """
    Fits a hyperplane to points by minimizing the sum of squared distances
    of the points from the hyperplane.
    
    Parameters:
    x (numpy.ndarray): A 2D array of shape (D, N) containing points in inhomogeneous coordinates.
                       D is the dimension, and N is the number of points.
    sigma (float, optional): Standard deviation of noise in the point coordinates.
    
    Returns:
    tuple: A tuple containing:
           - p (numpy.ndarray): A 1D array of shape (1, D+1) representing the hyperplane in
                                 homogeneous coordinates. The normal vector has a unit norm.
           - r (float): Sum of squared residuals.
           - Q (float): Probability that x fits the hyperplane model, assuming noise with standard
                        deviation sigma in x (chi-square fit). For a good fit, it should be Q > 0.1.
                        Returns None if sigma is not provided.
    """
    # Remove columns with all NaN values
    x = x[:, ~np.all(np.isnan(x), axis=0)]
    
    if x.size == 0:
        return (np.array([np.nan] * 4), np.nan, np.nan)
    
    D, N = x.shape
    c = np.sum(x, axis=1) / N
    x = x - c[:, None]
    
    u, s, vh = np.linalg.svd(x @ x.T)
    a = np.concatenate([np.zeros(D-1), [1]]) @ u.T
    p = np.concatenate([a, [-a @ c]])
    r = s[-1]
    
    Q = None
    if sigma is not None and N > D:
        chisq = r / sigma**2
        Q = 1 - gammainc((N-D)/2, chisq/2)
    
    return p, r, Q

def vgg_segments2lines(e, sigma, Qth):
    """
    Filters and merges edgel segments based on their fit to straight lines.
    
    Parameters:
    e (list): A list where each element is a 2D numpy array representing an edgel segment.
    sigma (float): Standard deviation of noise in the edgel coordinates.
    Qth (float): Threshold for the chi-square probability to decide if an edgel set belongs to a line segment.
    
    Returns:
    list: A list of edgel segments that are considered to fit straight lines.
    """
    E = []
    for n in range(len(e)):
        if e[n].shape[1] < 4:
            continue

        # Try to fit a straight line
        l, r, Q = vgg_hplanefit(e[n], sigma)

        if Q > Qth:  # It fits, include in the output list
            E.append(e[n])
        else:  # It doesn't fit; try to break it in two
            l = vgg_wedge(vgg_get_homg(e[n][:, [0, -1]]))  # Line through segment's end points
            dummy, i = np.abs(l @ vgg_get_homg(e[n])).max(0), np.abs(l @ vgg_get_homg(e[n])).argmax(0)
            y = [e[n][:, :i], e[n][:, i+1:]]  # Divide the chain at this point
            for j in range(2):  # Try to fit each piece to a line
                if y[j].shape[1] < 4:
                    continue
                l, r, Q = vgg_hplanefit(y[j], sigma)
                if Q > Qth:
                    E.append(y[j])

    # Merge neighboring line segments if possible
    cont = True
    while cont:
        cont = False
        n = 0
        while n < len(E) - 1:
            l, r, Q = vgg_hplanefit(np.hstack((E[n], E[n + 1])), sigma)
            if Q > Qth:
                E[n] = np.hstack((E[n], E[n + 1]))
                E.pop(n + 1)
                cont = True
            else:
                n += 1

    return E

def vgg_lineseg_from_x(e):
    """
    Fits a straight line to a set of 2D points, minimizing squared orthogonal distances,
    and finds endpoints of the line segment.
    
    Parameters:
    e (numpy.ndarray): A 2xN array of points.
    
    Returns:
    tuple: Endpoints of the line segment and optionally the covariance matrix.
    """
    C = np.vstack([e, np.ones(e.shape[1])])
    C = C @ C.T

    # Line fitted to points
    U, S, V = np.linalg.svd(C[:2, :2] - np.outer(C[:2, 2], C[:2, 2]) / C[2, 2], full_matrices=False)
    a = U[:, -1].T
    l = np.append(a, -a @ C[:2, 2] / C[2, 2])

    # Endpoints
    d = vgg_contreps(l[:2]).T @ e
    ab = np.array([-l[1], l[0]])
    cl = vgg_contreps(l)
    imin, imax = np.argmin(d), np.argmax(d)
    u = (cl @ np.append(ab, -ab @ e[:, imin]))[:2] / (cl @ np.append(ab, -ab @ e[:, imin]))[2]
    v = (cl @ np.append(ab, -ab @ e[:, imax]))[:2] / (cl @ np.append(ab, -ab @ e[:, imax]))[2]

    # Covariance matrix calculation if needed
    C[1:2, 1:2] = C[:2, :2] - np.outer(C[:2, 2], C[:2, 2]) / C[2, 2]
    C[:, 2] = np.hstack([C[:2, 2] / C[2, 2], C[2, 2]])

    return u, v, C

def vgg_linesegs_from_edgestrips(e, **kwargs):
    """
    Breaks edgel strips and approximates them by line segments.
    
    Parameters:
    e (list): A list of numpy arrays, each representing an edgel strip.
    **kwargs: Optional breaking/fitting parameters.
    
    Returns:
    tuple: Endpoints of the detected line segments and optionally the covariance matrices.
    """
    # Set default parameters
    opt = {
        'WormLength': kwargs.get('WormLength', 5),
        'WormThreshold': kwargs.get('WormThreshold', 0.5),  # Unused in this translation
        'ChisqSigma': kwargs.get('ChisqSigma', 0.45),
        'ChisqThreshold': kwargs.get('ChisqThreshold', 0.1)
    }

    np.set_printoptions(threshold=sys.maxsize)
    
    u, v, C = [], [], []
    for n in range(len(e)):
        strip = e[n][:2, :]  # Ensure only the first 2 rows are used, in case e is in homog coords
        if strip.shape[1] > 1:
            ij, ijn = vgg_chain2segments(strip, opt['WormLength'], 0.5)  # Threshold unused here
            c = []
            #print(strip)
            #print(ij)
            #print(ij.shape)
            for j in range(ij.shape[1]):
                print(f"{ij[0, j]}, {ij[1, j]+1}, {e[n][:, ij[1, j]:ij[0, j]+1]}")
                c.append(e[n][:, ij[0, j]:ij[1, j]+1])
            c = vgg_segments2lines(c, opt['ChisqSigma'], opt['ChisqThreshold'])
            for segment in c:
                u_end, v_end, Cj = vgg_lineseg_from_x(np.array(segment))
                u.append(u_end)
                v.append(v_end)
                C.append(vgg_vech(Cj))  # Assuming vgg_vech is implemented to handle conversion

    # Convert lists to numpy arrays for output
    u = np.array(u).T if u else np.array([])
    v = np.array(v).T if v else np.array([])
    C = np.array(C).T if C else np.array([])
    
    return u, v, C