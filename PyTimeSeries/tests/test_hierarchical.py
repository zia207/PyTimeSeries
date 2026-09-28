import numpy as np
import pandas as pd

import pytimeseries as pt


def test_summing_matrix_and_coherence():
    nodes, S = pt.build_summing_matrix(["a", "b", "c"])
    assert nodes == ["Total", "a", "b", "c"]
    assert S.shape == (4, 3)

    rng = np.random.default_rng(0)
    base = pd.DataFrame(rng.random((4, 5)) * 10, index=nodes)   # incoherent base
    for method in ["bottom_up", "top_down", "ols"]:
        rec = pt.reconcile(base, S, method=method)
        total = rec.loc["Total"].values
        kids = rec.loc[["a", "b", "c"]].values.sum(axis=0)
        assert np.allclose(total, kids)                        # coherent by construction


def test_reconciler_with_groups():
    r = pt.HierarchicalReconciler(["a", "b", "c", "d"],
                                  groups={"G1": ["a", "b"], "G2": ["c", "d"]})
    assert r.S.shape == (7, 4)                                 # Total + 2 groups + 4 bottom
