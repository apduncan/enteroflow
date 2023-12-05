"""Functions shared across the three NMF workflows (rank selection, 
L1 selection, decomposition). Most functions adapted from 
https://gitlab.inria.fr/cfrioux/enterosignature-paper/ by Anthony Duncan
(apduncan)"""

import numpy as np


def cosine_sim(x: np.ndarray, wh: np.ndarray) -> float:
    """Calculates the cosine similarity between two matrices

    :param x: Original matrix X on which NMF was performed
    :type x: np.ndarray
    :param wh: Product of NMF decomposition X ~ WH
    :type wh: np.ndarray
    :return: Cosine similarity
    :rtype: float
    """
    #TODO: Check that flattening to a vector is sensible way to get cosine
    # similarity for a matrix.
    # Use np.ravel for 1d view of matrices, to avoid extra memory from copies
    x_flat: np.array = np.ravel(x)
    wh_flat: np.array = np.ravel(wh)
    return wh_flat.dot(x_flat) / np.sqrt(
        x_flat.dot(x_flat) * wh_flat.dot(wh_flat)
    )

def evar(x: np.ndarray, wh: np.ndarray) -> float:
    """Calculates the explained variance metric between two matrices

    :param x: Original matrix X on which NMF was performed
    :type x: np.ndarray
    :param wh: Product of NMF decomposition X ~ WH
    :type wh: np.ndarray
    :return: Variance explained
    :rtype: float
    """
    return 1 - ( np.sum((np.array(x)-np.array(wh))**2) / np.sum(np.array(x)**2))

def rss_calc(x: np.ndarray, wh: np.ndarray) -> float:
    """Calculates the rss metric between two matrices

    :param x: Original matrix X on which NMF was performed
    :type x: np.ndarray
    :param wh: Product of NMF decomposition X ~ WH
    :type wh: np.ndarray
    :return: Residual sum of squares
    :rtype: float
    """
    return np.sum((np.array(x)-np.array(wh))**2)

def l2norm_calc(x: np.ndarray, wh: np.ndarray) -> float:
    """Calculates the l2 norm metric between two matrices

    :param x: Original matrix X on which NMF was performed
    :type x: np.ndarray
    :param wh: Product of NMF decomposition X ~ WH
    :type wh: np.ndarray
    :return: l2 norm
    :rtype: float
    """
    return np.sqrt(np.sum((np.array(x) - np.array(wh))**2))

def cut_in_four(m: np.ndarray, h: int = 3):
    """Takes a matrix and cuts it in 4 for cross validation following a ratio h
    e.g h = 3: the M1 submatrix for validation will have size 1/9 of the 
    matrix
    
    :param m: Matrix
    :type m: np.ndarray
    :param h: Ratio, defaults to 3
    :type h: int, optional
    :return: List of four matrices
    :rtype: List[np.ndarray]
    """
    nfeatures, nsamples = m.shape
    chunks_feat: int = nfeatures // h
    chunks_samp: int = nsamples // h
    thresholds_feat: int = [chunks_feat * i for i in range(1,h)]
    thresholds_samp: int = [chunks_samp * i for i in range(1,h)]
    m1: np.ndarray = m[0:thresholds_feat[0], 0:thresholds_samp[0]]
    m2: np.ndarray = m[0:thresholds_feat[0], thresholds_samp[0]:nsamples+1]
    m3: np.ndarray = m[thresholds_feat[0]:nfeatures+1, 0:thresholds_samp[0]]
    m4: np.ndarray = m[thresholds_feat[0]:nfeatures+1, 
                       thresholds_samp[0]:nsamples+1]
    return m1, m2, m3, m4
