"""Labels (week 2): mid-price direction N seconds ahead -- up, flat or down.

Horizons 1s, 10s, 60s. "Flat" = |move| below a per-pool threshold fit on training data
only, chosen so the three classes are roughly balanced.
"""


def make_labels(pool: str, horizon_s: int) -> None:
    raise NotImplementedError("week 2: labels")
