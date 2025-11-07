import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dt.api import app


@pytest.fixture()
def client():
    with app.test_client() as client:
        yield client


def make_job(job_id: str):
    return {
        "id": job_id,
        "deadline_ms": 1500,
        "stages": [
            {
                "id": "stage-1",
                "size_mb": 12,
                "resources": {"cpu_cores": 1, "mem_gb": 1},
                "allowed_formats": ["native", "wasm"],
            },
            {
                "id": "stage-2",
                "size_mb": 18,
                "resources": {"cpu_cores": 1, "mem_gb": 1},
                "allowed_formats": ["native", "wasm"],
            },
        ],
    }


def test_plan_returns_rich_stage_metrics(client):
    response = client.post("/plan", json={"job": make_job("job-1"), "dry_run": True})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True

    data = payload["data"]
    assert data["assignments"]
    per_stage = data["per_stage"]
    assert len(per_stage) == 2

    first_stage = per_stage[0]
    assert "score" in first_stage
    assert "reservation_id" in first_stage
    assert "compute_ms" in first_stage and first_stage["compute_ms"] >= 0


def test_plan_batch_reuses_plan_logic(client):
    jobs = [make_job("batch-1"), make_job("batch-2")]
    response = client.post("/plan_batch", json={"jobs": jobs, "dry_run": True})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True

    results = payload["data"]["results"]
    assert len(results) == 2
    assert results[0]["job_id"] == "batch-1"
    assert "score" in results[0]["per_stage"][0]


def test_plan_batch_propagates_errors(client):
    bad_job = {"id": "bad", "stages": []}
    response = client.post("/plan_batch", json={"jobs": [bad_job]})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert "job.stages is empty" in payload["error"]
