#!/usr/bin/env python
import logging
import pathlib
import pickle
from typing import Dict, List
from cvanmf import denovo
import click
import pandas as pd

@click.command()
@click.option("--matrix", "-i", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Input full matrix")
@click.option("--regu_res", "-r", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              readable=True),
              help="Results from regularisation selection, to pick alpha.")
@click.option("--random_starts", '-r', type=int, default=100,
              help="Number of random initialisations for random init methods")
@click.option("--seed", "-s", type=int, default=7297108,
              help="Random state seed.")
@click.option("--rank", "-k", type=int, required=True,
              help="Rank for this decomposition.")
@click.option("--l1_ratio", "-l", type=float, default=1.0,
              help="Ratio of L1 to L2 regularisation.")
@click.option("--max_iter", "-m", type=int, default=3000,
              help="Maximum iterations during each run of NMF.")
@click.option("--init", type=str, required=False,
              help="Initialisation method")
@click.option("--beta_loss", "-b",
              type=click.Choice(['kullback-leibler', 'frobenius',
                                 'itakura-saito'], case_sensitive=False),
              default="kullback-leibler",
              help="Beta-loss function to use during decomposition.")
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True,
              help="Show verbose logs.")

def cli(matrix: str,
        regu_res: str,
        random_starts: int,
        seed: int,
        rank: int,
        l1_ratio: float,
        max_iter: int,
        init: str,
        beta_loss: str,
        verbose: bool) -> None:
    """Produce a decomposition with heuristically selected alpha.

    :param matrix: Full matrix
    :param regu_res: Regularisation selection results
    :param seed: Random state seed
    :param rank: Rank of decomposition to run
    :param l1_ratio: Ratio of L1 to L2 regularisation
    :param max_iter: Maximum iterations to allow during NMF
    :param init: Initialisation method for W and H
    :param beta_loss: Beta-loss function for decomposition
    :param verbose: Activate verbose logging
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Load regularisation results, and matrix
    with open(regu_res, 'rb') as f:
        regu_dict: Dict[float, list[denovo.BicvResult]] = pickle.load(f)
    mat: pd.DataFrame = pd.read_csv(matrix, index_col=0, delimiter="\t")

    # Get heuristically determined best alpha
    best_alpha: float = denovo.suggest_alpha(regu_dict)
    # We don't need to scale this, will have already been scaled earlier if
    # desired

    logging.info(
        "Full Matrix Decomposition\n"
        "---------------------------\n"
        f"matrix        : {matrix}\n"
        f"seed          : {seed}\n"
        f"rank          : {rank}\n"
        f"max_iter      : {max_iter}\n"
        f"l1_ratio      : {l1_ratio}\n"
        f"alpha         : {best_alpha}\n"
        f"init          : {init}\n"
        f"beta_loss     : {beta_loss}"
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
            init=init,
            beta_loss=beta_loss
        )
    )

    # Output the best decomposition
    best_d: denovo.Decomposition = decompositions[rank][0]
    best_d.save("./regularised_model")

if __name__ == "__main__":
    cli()