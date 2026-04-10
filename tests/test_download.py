import pandas as pd

from llm_judge_audit.datasets import download


def test_get_items_from_source_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _DummyDataset:
        def to_pandas(self):
            return pd.DataFrame([{"x": 1}])

    def fake_load_dataset(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary failure")
        return _DummyDataset()

    monkeypatch.setattr(download, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(download.time, "sleep", lambda _s: None)

    df = download.get_items_from_source("foo/bar", "train", retries=3)
    assert not df.empty
    assert calls["n"] == 3


def test_get_items_from_source_all_fail_returns_empty_df(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        raise RuntimeError("always fails")

    monkeypatch.setattr(download, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(download.time, "sleep", lambda _s: None)

    df = download.get_items_from_source("foo/bar", "train", retries=2)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
