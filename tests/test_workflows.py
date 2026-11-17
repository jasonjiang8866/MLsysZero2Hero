import json
from pathlib import Path

import yaml

from mlsys_lab.budget import State
from mlsys_lab.results import Manifest
from mlsys_lab.worker import ExportManifest, Runtime

ROOT = Path(__file__).resolve().parents[1]


def test_published_schemas_match_runtime_models():
    for name, model in (
        ("e00-manifest", Manifest),
        ("budget-state", State),
        ("worker-runtime", Runtime),
        ("worker-export", ExportManifest),
    ):
        assert json.loads((ROOT / f"schemas/{name}.schema.json").read_text()) == (
            model.model_json_schema()
        )


def workflow(name):
    # BaseLoader preserves YAML's 'on' key instead of YAML 1.1's boolean conversion.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def test_cloud_workflow_is_manual_main_only_and_get_only():
    value = workflow("runpod-preflight.yml")
    assert set(value["on"]) == {"workflow_dispatch"}
    assert value["permissions"] == {"contents": "read"}
    job = value["jobs"]["observe"]
    assert "refs/heads/main" in job["if"] and "MLsysZero2Hero" in job["if"]
    run = "\n".join(s.get("run", "") for s in job["steps"])
    assert "mlsys-lab preflight" in run
    assert "controller" not in run and "--confirm-paid" not in run
    secret_steps = [s for s in job["steps"] if "RUNPOD_API_KEY" in s.get("env", {})]
    assert len(secret_steps) == 1
    assert "mlsys-lab preflight" in secret_steps[0]["run"]


def test_ci_has_no_cloud_secrets_and_pins_actions():
    value = workflow("ci.yml")
    assert "pull_request_target" not in value["on"]
    assert "secrets." not in (ROOT / ".github/workflows/ci.yml").read_text()
    for name in ("ci.yml", "runpod-preflight.yml", "build-worker.yml", "publish-worker.yml"):
        for job in workflow(name)["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    assert len(step["uses"].split("@")[1]) == 40


def test_build_has_no_secrets_and_publication_is_manual_main_only():
    build = workflow("build-worker.yml")
    assert "pull_request_target" not in build["on"]
    assert build["permissions"] == {"contents": "read"}
    assert "secrets." not in (ROOT / ".github/workflows/build-worker.yml").read_text()
    publish = workflow("publish-worker.yml")
    assert set(publish["on"]) == {"workflow_dispatch"}
    job = publish["jobs"]["publish"]
    assert "refs/heads/main" in job["if"] and "inputs.confirm_publish" in job["if"]
    assert job["permissions"] == {"contents": "read", "packages": "write"}
    text = (ROOT / ".github/workflows/publish-worker.yml").read_text()
    assert "RUNPOD_API_KEY" not in text
    assert "docker pull" in text and "SOURCE_SHA" in text
