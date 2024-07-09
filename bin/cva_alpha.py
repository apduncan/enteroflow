#!/usr/bin/env python
import logging
import pathlib
import pickle
import click
import numpy as np
from cvanmf import denovo

@click.command()
@click.option("--folds", "-f", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Shuffled and split matrix.")
@click.option("--seed", "-s", type=int, default=7297108,
              help="Random state seed.")
@click.option("--rank", "-k", type=int, required=True,
              help="Rank for this decomposition.")
@click.option("--alpha", "-a", type=float, required=True,
              help="Alpha parameter for regularisation")
@click.option("--scale/--no-scale", type=bool, default=True,
              help="Apply scaling to alpha values based on number of samples")
@click.option("--max_iter", "-m", type=int, default=3000,
              help="Maximum iterations during each run of NMF.")
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True,
              help="Show verbose logs.")
@click.option("--l1_ratio", "-l", type=float, default=1.0,
              help="Ratio of L1 to L2 regularisation.")
def cli(folds: str,
        seed: int,
        rank: int,
        alpha: float,
        scale: bool,
        max_iter: int,
        verbose: bool,
        l1_ratio: float) -> None:
    """Run bicross-validation for one shuffled and split matrix. 9 rearranged
    matrixes are made, and for each NMF is run once.

    :param folds: Path to shuffled and split matrix, saved in npz numpy format
    :type folds: str
    :param seed: Random state seed
    :type seed: int
    :param rank: Rank of decomposition to run
    :type rank: int
    :param scale: Rescale alpha to be proportional to number of 
    samples. 
    :type scale: bool
    :param max_iter: Maximum iterations to allow during NMF
    :type max_iter: iter
    :param verbose: Activate verbose logging
    :type verbose: bool
    :param l1_ratio: Ratio of L1 to L2 regularisation
    :type l1_ratio: float
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Load shuffled, split, and rearranged matrices
    shuff: denovo.BicvSplit = denovo.BicvSplit.load_npz(
        path=pathlib.Path(folds)
    )
    # Calculate a seed based on starting seed, plus rank and shuffle
    calc_seed: int = (seed * rank) + int(shuff.i)
    # Scaled alpha to number of samples 
    alpha_scale: float = alpha / (1.0 if not scale else shuff.shape[1])
    logging.info(scale, alpha_scale, shuff.shape)
    np.random.seed(calc_seed)
    logging.info(
        "Bicross Validation\n"
        "---------------------------\n"
        f"folds         : {folds}\n"
        f"seed          : {seed}\n"
        f"calc_seed     : {calc_seed}\n"
        f"rank          : {rank}\n"
        f"max_iter      : {max_iter}\n"
        f"l1_ratio      : {l1_ratio}\n"
        f"scale         : {scale}\n"
        f"alpha         : {alpha_scale}"
    )
    bicv_res: denovo.BicvResult = denovo.bicv(
        denovo.NMFParameters(
            x=shuff,
            rank=rank,
            seed=calc_seed,
            alpha=alpha_scale,
            l1_ratio=l1_ratio,
        )
    )
    with open("results.pickle", "wb") as f:
        pickle.dump(bicv_res, f)

if __name__ == "__main__":
    cli()