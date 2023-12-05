#!/usr/bin/env python
"""Split a matrix ready for NMF bi-cross-validation

Shuffle and split matrix X into 9 submatrices X_1 ... X_9 for use in 
bi-cross-validation."""

from ast import Dict
from typing import Iterable, List
import click
import numpy as np
import pandas as pd

def split_matrix(
        matrix: pd.DataFrame, seed: int, i: int) -> Iterable[pd.DataFrame]:
    """Shuffle and split a matrix for NMF bi-cross-validation. For 
    reproducibility, use seed + i as seed to random Generator. 
    Adapted from bicv_rank.py at 
    https://gitlab.inria.fr/cfrioux/enterosignature-paper. The procedure is
    defined for more than a 3 x 3 split, but this pipeline uses only 9-fold, 
    so we fix at 3 x 3 in this function.

    :param matrix: Input matrix
    :type matrix: str
    :param seed: Random seed for shuffling
    :type seed: int
    :param i: This is the i-th shuffle
    :type i: int
    :return: Matrix X splits into 9 dataframe X_1 ... X_9
    :rtype: Iterable[pd.DataFrame]
    """

    m_shuf: np.ndarray = np.matrix.copy(matrix.to_numpy())
    # Shuffle
    random_generator: np.random.Generator = np.random.default_rng(seed=seed+i)
    random_generator.shuffle(m_shuf, axis=0)
    random_generator.shuffle(m_shuf, axis=1)
    # Cut the h x h submatrices
    n_features, n_samples = m_shuf.shape
    chunks_feat = n_features // 3
    #remainings_feat = nfeatures % h
    chunks_samp = n_samples // 3
    #remainings_samp = nsamples % h
    thresholds_feat = [chunks_feat * i for i in range(1, 3)]
    thresholds_feat.insert(0,0)
    thresholds_feat.append(n_features+1)
    thresholds_samp = [chunks_samp * i for i in range(1, 3)]
    thresholds_samp.insert(0,0)
    thresholds_samp.append(n_samples+1)
    # Return the 9 matrices
    matrix_nb = [i for i in range(1, 3*3 + 1)]
    all_sub_matrices = [None] * 9
    done = 0
    row = 0
    col = 0
    while row < 3:
        while col < 3:
            done += 1
            all_sub_matrices[done-1] = m_shuf[
                thresholds_feat[row]:thresholds_feat[row+1],
                thresholds_samp[col] : thresholds_samp[col+1]
            ]
            col += 1
        row += 1
        col = 0
    assert(len(all_sub_matrices) == len(matrix_nb))
    return all_sub_matrices

def concat_3x3_mx(
    m_list: np.ndarray,
    row1: List[int],
    row2: List[int],
    row3: List[int]) -> np.ndarray:
    """Concatenates submatrices based on the order given in arguments. Adapted
    from https://gitlab.inria.fr/cfrioux/enterosignature-paper/.

    :param m_dict: List of split matrices, from top left to bottom right
    :type m_dict: List[np.ndarray]
    :param row1: Order of parts in row 1 of rearranged matrix
    :type row1: List[int]
    :param row2: Order of parts in row 2 of rearranged matrix
    :type row2: List[int]
    :param row3: Order of parts in row 3 of rearranged matrix
    :type row3: List[int]
    :return: Matrix with split block rearranged
    :rtype: np.ndarray
    """
    r1 = np.concatenate(
        (m_list[row1[0]-1], m_list[row1[1]-1], m_list[row1[2]-1]), axis=1)
    r2 = np.concatenate(
        (m_list[row2[0]-1], m_list[row2[1]-1], m_list[row2[2]-1]), axis=1)
    r3 = np.concatenate(
        (m_list[row3[0]-1], m_list[row3[1]-1], m_list[row3[2]-1]), axis=1)
    m = np.concatenate((r1, r2, r3), axis = 0)
    return m

def bicv_3x3(sm_list: List[np.ndarray]) -> List[np.ndarray]:
    """Starting from 9 submatrices, rearrange them to create 9 matrices used 
    for biCV. Adapated from 
    https://gitlab.inria.fr/cfrioux/enterosignature-paper/

    :param sm_list: The shuffled submatrices
    :type sm_list: List[np.ndarray]
    :return: Nine matrices made up of rearrangements of the submatrices
    :rtype: List[np.ndarray]
    """
    all_mx = {}
    # m1 (all rows) 123 - 456 - 789
    all_mx[1] = concat_3x3_mx(sm_list, [1,2,3], [4,5,6], [7,8,9])    
    # m2 312 - 645 - 978
    all_mx[2] = concat_3x3_mx(sm_list, [3,1,2], [6,4,5], [9,7,8]) 
    # m3 231 - 564 - 897
    all_mx[3] = concat_3x3_mx(sm_list, [2,3,1], [5,6,4], [8,9,7]) 
    # m4 789 - 123 - 456
    all_mx[4] = concat_3x3_mx(sm_list, [7,8,9], [1,2,3], [4,5,6])  
    # m5 456 - 789 - 123
    all_mx[5] = concat_3x3_mx(sm_list, [4,5,6], [7,8,9], [1,2,3])  
    # m6 645 - 978 - 312
    all_mx[6] = concat_3x3_mx(sm_list, [6,4,5], [9,7,8], [3,1,2])  
    # m7 978 - 312 - 645
    all_mx[7] = concat_3x3_mx(sm_list, [9,7,8], [3,1,2], [6,4,5])  
    # m8 897 - 231 - 564
    all_mx[8] = concat_3x3_mx(sm_list, [8,9,7], [2,3,1], [5,6,4])  
    # m9 564 - 897 - 231
    all_mx[9] = concat_3x3_mx(sm_list, [5,6,4], [8,9,7], [2,3,1])  
    return list(all_mx.values())

@click.command()
@click.option("--matrix", "-m", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Input matrix in tab-separated format")
@click.option("--seed", "-s", type=int, default=7297108,
              help="Random state seed")
@click.option("--iteration", "-i", type=int, required=True,
              help="Iteration number, i.e. this is the i-th split")
def cli(matrix: str, seed: int, iteration: int) -> None:
    """Shuffle and split a matrix for NMF 9-fold bi-fold-cross validation. 
    Seed for the shuffle is seed+i to ensure reproducibility. The output is 
    the shuffled and split matrix rearranged into 9 matrices.

    :param matrix: Input matrix location
    :type matrix: str
    :param seed: Random seed
    :type seed: int
    :param i: This is the i-th split
    :type i: int
    """
    folds: List[np.ndarray] = split_matrix(
        pd.read_csv(matrix, sep="\t", index_col=0),
        seed=seed,
        i=iteration
    )
    mx: List[np.ndarray] = bicv_3x3(folds)
    np.savez("submats.npz", *mx)

if __name__ == "__main__":
    cli()
