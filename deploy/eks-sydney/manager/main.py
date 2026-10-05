import requests
from fastapi import FastAPI
from pydantic import BaseModel

import storage

app = FastAPI()

MODEL_ENDPOINTS = {
    "ppe-compliance": "http://ppe-compliance/invocations",
    "running-detection": "http://running-detection/invocations",
    "climb-detection": "http://climb-detection/invocations",
    "phone-usage-detection": "http://phone-usage-detection/invocations",
    "sleep-detection": "http://sleep-detection/invocations",
    "videomae-aggression": "http://videomae-aggression/invocations",
}

TIMEOUT = 90


class CallingJSON(BaseModel):
    job_id: str
    footage_s3_key: str
    models_requested: list[str]


def _empty_media():
    return {"annotated_video_s3_key": None, "annotated_image_s3_key": None}


def skipped_block(name):
    return {
        "model_id": name, "model_version": None, "status": "SKIPPED",
        "processing_time_ms": None, "output_media": _empty_media(),
        "detections": [], "error": None,
    }


def failed_block(name, code, message):
    return {
        "model_id": name, "model_version": None, "status": "FAILED",
        "processing_time_ms": None, "output_media": _empty_media(),
        "detections": [], "error": {"code": code, "message": message},
    }


def call_model(name, job_id, footage_s3_key):
    url = MODEL_ENDPOINTS.get(name)
    if url is None:
        return skipped_block(name)
    try:
        resp = requests.post(
            url,
            json={"job_id": job_id, "footage_s3_key": footage_s3_key},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return failed_block(name, "MODEL_TIMEOUT", f"exceeded {TIMEOUT}s")
    except Exception as e:
        return failed_block(name, "MODEL_ERROR", str(e))


def job_status(results):
    statuses = {r["status"] for r in results}
    if "FAILED" not in statuses:
        return "COMPLETED"
    if "SUCCESS" in statuses:
        return "PARTIAL"
    return "FAILED"


@app.post("/analyze")
def analyze(job: CallingJSON):
    results = [
        call_model(name, job.job_id, job.footage_s3_key)
        for name in job.models_requested
    ]
    status = job_status(results)

    response = {
        "job_id": job.job_id,
        # Persist the input contract with the result. The review dashboard must
        # still be able to open the source video when every model invocation
        # fails and therefore produces no output media.
        "footage_s3_key": job.footage_s3_key,
        "models_requested": job.models_requested,
        "status": status,
        "results": results,
        "error": None if status != "FAILED" else {
            "code": "ALL_MODELS_FAILED",
            "message": "no model produced a successful result",
        },
    }
    key = f"{storage.job_prefix(job.job_id)}/response.json"
    response["response_s3_key"] = key
    storage.save_json(key, response)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}
