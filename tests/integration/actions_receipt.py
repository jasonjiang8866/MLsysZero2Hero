"""Check a freshly uploaded CI fixture against GitHub using a local-only journal."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from runpod_ci_broker.ledger import SQLiteStore

from mlsys_lab.artifact import github_json, uploaded_artifact
from mlsys_lab.controller import Journal, digest, receipt


def main():
    env = os.environ
    fixture = Path("out/receipt-fixture")
    with TemporaryDirectory() as temp:
        store = SQLiteStore(Path(temp) / "receipt.sqlite3")
        journal = Journal(store, env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"])
        journal.save(
            phase="collected",
            job={"source_sha": env["RUN_HEAD_SHA"]},
            manifest_digest=digest((fixture / "manifest.json").read_bytes()),
        )
        artifact_id = env["ARTIFACT_ID"]
        archive_digest = env["ARCHIVE_DIGEST"]
        artifact = uploaded_artifact(journal.run, journal.attempt, artifact_id, fetch=github_json)
        receipt(journal, artifact, artifact_id, archive_digest, fixture)
        saved = Journal(store, journal.run, str(journal.number)).record
        assert saved["artifact_id"] == int(artifact_id)
        assert saved["archive_digest"] == "sha256:" + archive_digest.removeprefix("sha256:")


if __name__ == "__main__":
    main()
