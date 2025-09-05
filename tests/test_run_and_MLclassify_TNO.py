from sbdynt.horizons_api import query_sb_from_jpl
from sbdynt.machine_learning import run_and_MLclassify_TNO

target_ids = [
    "2014 FE72",
    "2015 VP172",
    # "2012 HK85",
    # "2023 PF5",
    # "2005 UQ513",
    # "1997 CU29",
    # "2020 KP55",
    # "2025 OO1",
]


def test_run_and_MLclassify_TNO():
    for designation in target_ids:
        _, tno_class, _ = run_and_MLclassify_TNO(
            des=designation,
            datadir="test-data.local",
            logfile="screen",  # type: ignore
            deletefile=True,
        )
        assert tno_class is not None, "Classification failed"


def test_query_sb_from_jpl():
    for designation in target_ids:
        body = query_sb_from_jpl(designation)
        assert body is not None, "Query failed"


if __name__ == "__main__":
    test_query_sb_from_jpl()
