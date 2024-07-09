#!/usr/bin/env python
import logging
import pathlib
import pickle
from typing import Dict, List
from cvanmf import denovo
import click
import pandas as pd

@click.command()
@click.option("--input", "-i", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Full matrix")
@click.option("--regu_res", "-r", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Results from regularisation selection, to pick alpha.")
@click.option("--seed", "-s", type=int, default=7297108,
              help="Random state seed.")
@click.option("--rank", "-k", type=int, required=True,
              help="Rank for this decomposition.")
@click.option("--max_iter", "-m", type=int, default=3000,
              help="Maximum iterations during each run of NMF.")
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True,
              help="Show verbose logs.")
@click.option("--l1_ratio", "-l", type=float, default=1.0,
              help="Ratio of L1 to L2 regularisation.")
@click.option("--random_starts", '-r', type=int, default=100,
              help="Number of random initialisations for random init methods")
@click.option("--init", type=str, required=False,
              help="Initialisation method")
def cli(input: str,
        regu_res: str,
        seed: int,
        rank: int,
        max_iter: int,
        verbose: bool,
        l1_ratio: float,
        random_starts: int,
        init: str) -> None:
    """Produce a decomposition with heuristically selected alpha.

    :param input: Full matrix
    :param regu_res: Regularisation selection results
    :param seed: Random state seed
    :param rank: Rank of decomposition to run
    :param scale: Rescale alpha to be proportional to number of 
    samples. 
    :param max_iter: Maximum iterations to allow during NMF
    :param verbose: Activate verbose logging
    :param l1_ratio: Ratio of L1 to L2 regularisation
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Load regularisation results, and matrix
    with open(regu_res, 'rb') as f:
        regu_dict: Dict[float, list[denovo.BicvResult]] = pickle.load(f)
    mat: pd.DataFrame = pd.read_csv(input, index_col=0, delimiter="\t")

    # Get heuristically determined best alpha
    best_alpha: float = denovo.suggest_alpha(regu_dict)
    # We don't need to scale this, will have already been scaled.

    logging.info(
        "Full Matrix Decomposition\n"
        "---------------------------\n"
        f"input         : {input}\n"
        f"seed          : {seed}\n"
        f"rank          : {rank}\n"
        f"max_iter      : {max_iter}\n"
        f"l1_ratio      : {l1_ratio}\n"
        f"alpha         : {best_alpha}"
    )

    decompositions: Dict[int, List[denovo.Decomposition]] = (
        denovo.decompositions(
            x=mat,
            ranks=[rank],
            random_starts=random_starts,
            top_n=1,
            seed=seed,
            alpha=best_alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            init=init
        )
    )

    # Output the best decomposition
    best_d: denovo.Decomposition = decompositions[rank][0]
    best_d.save("./regularised_model")

if __name__ == "__main__":
    cli()