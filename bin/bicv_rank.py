#!/usr/bin/env python
from collections import defaultdict
import json
import logging
from typing import List
import click
import numpy as np
from numpy.lib.npyio import NpzFile
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.decomposition._nmf import _beta_divergence
from nmf_lib import *

def run_cv(mx9: np.ndarray,
           rank: int,
           maxiter: int = 2000,
           nruns: int = 10) -> None:
    """Runs cross validation for a set of 9 matrices corresponding to shuffled
    versions of the original matrix in which each 1/9 fold becomes the 
    validation set. The NMF algorithm is applied to all 9 matrices for a 
    given rank. For each combination of matrix and k, the NMF algorithm is run
    nruns times. For each run, the values of reconstruction error, cosine 
    similarity, L2 norm, RSS, and explained variance are calculated and returned
    in dictionaries. Adapted from 
    https://gitlab.inria.fr/cfrioux/enterosignature-paper/
    """
    # Initialise dictionaries to hold measurements from each run.
    res, rss, reco_error, cosine, l2_norm = tuple(
        defaultdict(lambda: dict(A=list(), B=list(), C=list(), D=list())) 
        for _ in range(5)
    )
    for i, mx in enumerate(mx9):
        M1, M2, M3, M4 = cut_in_four(mx, 3)
        for j in range(0, nruns):
            logpref = f"[Matrix {i+1}/9, Run {j+1}/{nruns}]"
            logging.info("%s Fit D", logpref)
            # Step 1, NMF for M_d
            model_D = NMF(n_components=rank,
                        init="nndsvdar",
                        verbose=False,
                        solver="mu",
                        max_iter=maxiter,
                        random_state = None,
                        alpha_W = 0,
                        alpha_H = 0,
                        beta_loss = "kullback-leibler")
            H_d = np.transpose(model_D.fit_transform(np.transpose(M4)))
            W_d = np.transpose(model_D.components_)
            Md_calc = W_d.dot(H_d)
            #res[rank]["D"] = {"M":M4, "M'":Md_calc, "evar":evar_M_d}
            res[i]["D"].append(evar(M4, Md_calc))
            reco_error[i]["D"].append(model_D.reconstruction_err_)
            rss[i]["D"].append(rss_calc(M4, Md_calc))
            l2_norm[i]["D"].append(l2norm_calc(M4, Md_calc))
            cosine[i]["D"].append(cosine_sim(M4, Md_calc))
            #print(f"k = {rank} - Evar M_d - {evar_M_d}")
            # step 2, get W_a using M_b
            logging.info("%s Get W_a", logpref)
            W_a, H_d_t, n_iter = non_negative_factorization(M2, 
                                                n_components=rank,
                                                init='custom',
                                                verbose=False,
                                                solver="mu",
                                                max_iter=maxiter,
                                                random_state=None,
                                                alpha_W=0,
                                                alpha_H=0,
                                                beta_loss = "kullback-leibler",
                                                update_H=False,
                                                H=H_d)

            Mb_calc = np.dot(W_a, H_d)
            #res[rank]["B"] = {"M":M2, "M'":Mb_calc, "evar":evar_M_b}
            res[i]["B"].append(evar(M2, Mb_calc))
            # print shapes
            # print(f"shape M2: {M2.shape} -- shape W_a: {W_a.shape} -- shape H_d: {H_d.shape}")
            reco_error[i]["B"].append(_beta_divergence(np.array(M2), np.array(W_a), np.array(H_d), "kullback-leibler",
                                                square_root=True))
            rss[i]["B"].append(rss_calc(M2, Mb_calc))
            l2_norm[i]["B"].append(l2norm_calc(M2, Mb_calc))
            cosine[i]["B"].append(cosine_sim(M2, Mb_calc))
            # Step 3, get H_a using M_c
            logging.info("%s Get H_a", logpref)
            H_a, W_d_t, n_iter = non_negative_factorization(M3.T,
                                                n_components=rank,
                                                init='custom',
                                                verbose=False,
                                                solver="mu",
                                                max_iter=maxiter,
                                                random_state=None,
                                                alpha_H=0,
                                                alpha_W=0,
                                                beta_loss="kullback-leibler",
                                                update_H=False,
                                                H=W_d.T)


            Mc_calc = np.dot(W_d, H_a.T)
            res[i]["C"].append(evar(M3, Mc_calc))
            reco_error[i]["C"].append(_beta_divergence(np.array(M3), np.array(W_d), np.array(H_a).T, "kullback-leibler",
                                                square_root=True))
            rss[i]["C"].append(rss_calc(M3, Mc_calc))
            l2_norm[i]["C"].append(l2norm_calc(M3, Mc_calc))
            cosine[i]["C"].append(cosine_sim(M3, Mc_calc))
            # step 4, calculate error for M_a
            logging.info("%s Calculate error for M_a", logpref)
            Ma_calc = np.dot(W_a, H_a.T)
            res[i]["A"].append(evar(M1, Ma_calc))
            reco_error[i]["A"].append(_beta_divergence(np.array(M1), np.array(W_a), np.array(H_a).T, "kullback-leibler",
                                                square_root=True))
            rss[i]["A"].append(rss_calc(M1, Ma_calc))
            l2_norm[i]["A"].append(l2norm_calc(M1, Ma_calc))
            cosine[i]["A"].append(cosine_sim(M1, Ma_calc))
    return res, rss, reco_error, cosine, l2_norm


@click.command()
@click.option("--folds", "-f", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Shuffled and split matrix.")
@click.option("--seed", "-s", type=int, default=7297108,
              help="Random state seed.")
@click.option("--rank", "-k", type=int, required=True,
              help="Rank for this decomposition.")
@click.option("--num_runs", "-n", type=int, default=100,
              help="Number of times to run the NMF algorithm.")
@click.option("--max_iter", "-m", type=int, default=3000,
              help="Maximum iterations during each run of NMF.")
@click.option("--shuffle_num", "-s", type=int, required=True,
              help="Which shuffle number of the original matrix this is.")
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True,
              help="Show verbose logs.")
def cli(folds: str,
        seed: int,
        rank: int,
        num_runs: int,
        max_iter: int,
        shuffle_num: int,
        verbose: bool) -> None:
    """Run bicross-validation for one shuffled and split matrix. 9 rearranged
    matrixes are made, and for each NMF is run on a portion num_runs times.

    :param folds: Path to shuffled and split matrix, saved in npz numpy format
    :type folds: str
    :param seed: Random state seed
    :type seed: int
    :param rank: Rank of decomposition to run
    :type rank: int
    :param num_runs: Number of runs per matrix
    :type num_runs: int
    :param max_iter: Maximum iterations to allow during NMF
    :type max_iter: iter
    :param shuffle_num: Which shuffle of the matrix this is
    :type int:
    :param verbose: Activate verbose logging
    :type bool:
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Calculate a seed based on starting seed, plus rank and shuffle
    calc_seed: int = (seed * rank) + shuffle_num
    np.random.seed(calc_seed)
    logging.info(
        "Bicross Validation\n"
        "---------------------------\n"
        f"folds         : {folds}\n"
        f"seed          : {seed}\n"
        f"calc_seed     : {calc_seed}\n"
        f"rank          : {rank}\n"
        f"num_runs      : {num_runs}\n"
        f"max_iter      : {max_iter}\n"
        f"shuffle_num   : {shuffle_num}\n"
    )
    # Load shuffled, split, and rearranged matrices
    arrays: NpzFile
    logging.info("Loading matrices")
    with np.load(folds) as arrays:
        arr_list: List[np.ndarray] = [arrays[f] for f in arrays.files]
        res, rss, reco_error, cosine, l2_norm = run_cv(
            mx9         = arr_list,
            rank        = rank,
            maxiter     = max_iter,
            nruns       = num_runs
        )
    # Output these results as a JSON file
    logging.info("Bicross validation complete, writing results")
    out_res = {name: vals for name, vals in [
        ('res', res), ('rss', rss), ('reco_error', reco_error),
        ('cosine', cosine), ('l2norm', l2_norm)
    ]}
    # Add the params used for this run, might be useful
    out_res['params'] = dict(
        folds = folds,
        seed = seed,
        rank = rank,
        num_runs = num_runs,
        max_iter = max_iter,
        shuffle_num = shuffle_num
    )
    with open("results.json", "wt", encoding="utf-8") as f:
        json.dump(out_res, f, indent=4)

if __name__ == "__main__":
    cli()