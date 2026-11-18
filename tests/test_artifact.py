from urllib.error import HTTPError

import pytest

from mlsys_lab.artifact import uploaded_artifact


def test_lookup_waits_for_exact_id_in_original_run():
    observations = iter(
        [
            HTTPError("", 404, "missing", {}, None),
            {"artifacts": [{"id": 11}]},
            {"artifacts": [{"id": 12}]},
        ]
    )
    waits = []

    def fetch(url):
        assert url.endswith("/runs/123/artifacts?name=E00-123-1")
        entry = next(observations)
        if isinstance(entry, HTTPError):
            raise entry
        return entry

    assert uploaded_artifact("123", "E00-123-1", "12", fetch=fetch, sleep=waits.append) == {
        "id": 12
    }
    assert waits == [1, 2]


def test_absent_artifact_stays_unverified():
    with pytest.raises(ValueError, match="unavailable"):
        uploaded_artifact(
            "123", "E00-123-1", "12", fetch=lambda _: {"artifacts": []}, sleep=lambda _: None
        )
