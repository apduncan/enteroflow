#!/usr/bin/env python

from collections import defaultdict
import json
import os
from typing import Any, Dict, List
import click


@click.command()
@click.argument("files", nargs=-1,
                type=click.Path(exists=True, file_okay=True, dir_okay=False,
                                readable=True))
def cli(files: List[str]) -> None:
    """Join the reesults from multiple runs into a single dictionary, in the 
    form expected by the jupyter notebooks from 
    https://gitlab.inria.fr/cfrioux/enterosignature-paper/"""

    # We start not knowing how many shuffles were performed, so we will
    # store in a dict then convert
    measures: Dict = dict(
        res=defaultdict(dict), rss=defaultdict(dict),
        reco_error=defaultdict(dict), cosine=defaultdict(dict),
        l2norm=defaultdict(dict), sparsity=defaultdict(dict)
    )
    # Target structure is List[Dict[fold][rank][alpha][A-D]]
    # We'll build with dict at the top level rather than list, then convert
    # though, as it's easier when we're getting shuffles out of order
    for path in files:
        with open(path, "rt", encoding="utf-8") as f:
            res: Dict[int, Any] = json.load(f)
            shuffle_i: int = res['params']['shuffle_num']
            rank: int = res['params']['rank']
            # Our dict will be key = shuffle, a second dict with 
            # key = measurment
            for meas, vals in res.items():
                # Ignore params item
                if meas == "params":
                    continue
                if shuffle_i not in measures[meas]:
                    measures[meas][shuffle_i] = defaultdict(dict)
                # Add values for each of mx9 for this shuffe+rank combo
                for alpha, alpha_vals in vals.items():
                    for mx, mx_vals in alpha_vals.items():
                        if mx not in measures[meas][shuffle_i]:
                            measures[meas][shuffle_i][mx] = dict()
                        if rank not in measures[meas][shuffle_i][mx]:
                            measures[meas][shuffle_i][mx][rank] = dict()
                        if alpha not in measures[meas][shuffle_i][mx][rank]:
                            measures[meas][shuffle_i][mx][rank][alpha] = dict()
                        measures[meas][shuffle_i][mx][rank][alpha] = mx_vals
    # Output each measure
    for meas, vals in measures.items():
        meas_f: str = "evar" if meas == "res" else meas
        outfile: str = f"biCV_regu_{meas_f}.json"
        # Convert dict to list
        outlist: List = [vals[i] for i in sorted(vals.keys())]
        with open(outfile, "wt", encoding="utf-8") as f:
            json.dump(outlist, f, indent=4)

if __name__ == "__main__":
    cli()
