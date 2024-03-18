#!/usr/bin/env python

import itertools
import logging
from typing import Dict, List
import pickle

from cvanmf import denovo
import click


@click.command()
@click.argument("files", nargs=-1,
                type=click.Path(exists=True, file_okay=True, dir_okay=False,
                                readable=True))
def cli(files: List[str]) -> None:
    """Join the results from multiple runs into a single dictionary, in the 
    form expected by the jupyter notebooks from 
    https://gitlab.inria.fr/cfrioux/enterosignature-paper/"""

    res: List[denovo.BicvResult] = [pickle.load(open(f, 'rb')) for f in files]

    results: List[denovo.BicvResult] = sorted(
        res,
        key=lambda x: x.parameters.rank
    )

    # Collect results from the ranks into lists, and place in a dictionary
    # with key = rank
    grouped_results: Dict[int, List[denovo.BicvResult]] = {
        rank_i: list(list_i) for rank_i, list_i in
        itertools.groupby(results, key=lambda z: z.parameters.rank)
    }

    # Validate that there are the same number of results for each rank.
    # Decomposition shouldn't silently fail, but best not to live in a
    # world of should. Currently deciding to warn user and still return
    # results.
    if len(set(len(y) for y in grouped_results.values())) != 1:
        logging.error(("Uneven number of results returned for each rank, "
                       "some rank selection iterations may have failed."))

    with open("rank_combined.pickle", "wb") as f:
        pickle.dump(grouped_results, f)
    
    (denovo.BicvResult.results_to_table(grouped_results)
     .to_csv('rank_selection.tsv', sep="\t"))

if __name__ == "__main__":
    cli()
