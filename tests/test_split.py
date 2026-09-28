"""The holdout split is deterministic and ~10% of entities."""
from split import assign_roles


def test_split_is_deterministic_and_about_ten_percent():
    ids = [f"S1-{i}" for i in range(50_000)]
    a, b = assign_roles(ids, subset_size=10_000), assign_roles(ids, subset_size=10_000)
    assert (a == b).all()
    share = (a == "holdout").mean()
    assert 0.09 < share < 0.11
    assert (a == "train").sum() == 10_000
