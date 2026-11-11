"""Walk-forward folds: training never sees a label that reaches into its test week."""

import pandas as pd

from flowcast.models.walkforward import FOLDS, in_test, in_train, ms


def test_folds_are_contiguous_and_cover_september():
    assert FOLDS[0][0] == "2025-09-01" and FOLDS[-1][1] == "2025-10-01"
    assert all(a[1] == b[0] for a, b in zip(FOLDS[:-1], FOLDS[1:], strict=True))


def test_training_labels_end_before_the_test_week():
    start = "2025-09-08"
    ts = pd.Series(range(ms("2025-09-07"), ms("2025-09-09"), 1000)) + 1000  # second ends
    for h in (1, 10, 60):
        tr = ts[in_train(ts, "2025-08-02", start, h)]
        assert (tr + h * 1000 <= ms(start)).all()
        assert tr.max() == ms(start) - h * 1000  # nothing extra purged


def test_train_and_test_rows_never_overlap():
    ts = pd.Series(range(ms("2025-08-30"), ms("2025-09-16"), 1000)) + 1000
    for start, end in FOLDS[:2]:
        tr, te = ts[in_train(ts, "2025-08-02", start, 1)], ts[in_test(ts, start, end)]
        assert tr.max() < te.min()
        assert te.min() == ms(start) + 1000 and te.max() == ms(end)
