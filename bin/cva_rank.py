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
@click.option("--max_iter", "-m", type=int, default=3000,
              help="Maximum iterations during each run of NMF.")
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True,
              help="Show verbose logs.")
def cli(folds: str,
        seed: int,
        rank: int,
        max_iter: int,
        verbose: bool) -> None:
    """Run bicross-validation for one shuffled and split matrix. 9 rearranged
    matrixes are made, and for each NMF is run once.

    :param folds: Path to shuffled and split matrix, saved in npz numpy format
    :type folds: str
    :param seed: Random state seed
    :type seed: int
    :param rank: Rank of decomposition to run
    :type rank: int
    :param max_iter: Maximum iterations to allow during NMF
    :type max_iter: iter
    :param verbose: Activate verbose logging
    :type bool:
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Load shuffled, split, and rearranged matrices
    shuff: denovo.BicvSplit = denovo.BicvSplit.load_npz(
        path=pathlib.Path(folds)
    )
    # Calculate a seed based on starting seed, plus rank and shuffle
    calc_seed: int = (seed * rank) + int(shuff.i)
    np.random.seed(calc_seed)
    logging.info(
        "Bicross Validation\n"
        "---------------------------\n"
        f"folds         : {folds}\n"
        f"seed          : {seed}\n"
        f"calc_seed     : {calc_seed}\n"
        f"rank          : {rank}\n"
        f"max_iter      : {max_iter}\n"
    )
    bicv_res: denovo.BicvResult = denovo.bicv(
        denovo.NMFParameters(
            x=shuff,
            rank=rank,
            seed=calc_seed,
            alpha=0.0,
            l1_ratio=0.0
        )
    )
    with open("results.pickle", "wb") as f:
        pickle.dump(bicv_res, f)

if __name__ == "__main__":
    cli()