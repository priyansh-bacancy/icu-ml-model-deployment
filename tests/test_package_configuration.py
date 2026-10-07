"""Offline checks for safe deployment templates and copied Python syntax."""

import ast
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("fall-detection", "restricted-zone", "mmc-vehicle-classification")


def test_all_recovered_python_files_parse() -> None:
    for model in MODELS:
        for source in (ROOT / "models" / model).rglob("*.py"):
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_services_and_deployments_agree_without_fixed_nodes() -> None:
    for model in MODELS:
        documents = list(
            yaml.safe_load_all(
                (ROOT / "deploy" / "eks-recovered" / f"{model}.yaml").read_text()
            )
        )
        account, service, deployment = documents
        spec = deployment["spec"]
        pod = spec["template"]["spec"]
        container = pod["containers"][0]

        assert account["kind"] == "ServiceAccount"
        assert service["metadata"]["name"] == model
        assert service["spec"]["selector"] == spec["selector"]["matchLabels"]
        assert spec["template"]["metadata"]["labels"] == spec["selector"]["matchLabels"]
        assert pod["serviceAccountName"] == account["metadata"]["name"]
        assert spec["replicas"] == 0
        assert "nodeName" not in pod and "nodeSelector" not in pod
        assert container["resources"]["requests"]["nvidia.com/gpu"] == 1
        assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
        assert container["startupProbe"]["httpGet"]["path"] == "/ping"
        assert container["image"].startswith("example.invalid/")
        assert container["image"].count("sha256:") == 1


def test_example_contracts_match_model_names() -> None:
    for model in MODELS:
        response = json.loads(
            (ROOT / "models" / model / "response-contract.example.json").read_text()
        )

        assert response["model_id"] == model
        assert len(response) == 9
        confidence = response["detections"][0]["instances"][0]["confidence"]
        assert isinstance(confidence, (int, float)) and 0 <= confidence <= 1
        assert isinstance(response["output_media"]["annotated_image_s3_keys"], list)


def test_dockerfiles_only_copy_runtime_and_named_checkpoints() -> None:
    for model in MODELS:
        recipe = (ROOT / "models" / model / "Dockerfile.gpu").read_text()

        assert 'ENTRYPOINT ["python", "/app/entrypoint.py"]' in recipe
        assert "COPY . " not in recipe
        assert "COPY weights/" in recipe
        assert "uv pip install" in recipe
