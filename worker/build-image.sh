#!/usr/bin/env bash
set -euo pipefail
source_sha="${1:?full source SHA required}"
[[ "$source_sha" =~ ^[a-f0-9]{40}$ ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$source_sha" ]] || exit 2
git diff --exit-code HEAD -- pyproject.toml uv.lock README.md LICENSE src worker
[[ -z "$(git ls-files --others --exclude-standard -- src worker)" ]] || exit 2
base_tag='python:3.12.12-slim-bookworm'
base_digest="$(docker buildx imagetools inspect "$base_tag" --format '{{json .Manifest.Digest}}' | tr -d '"')"
[[ "$base_digest" =~ ^sha256:[a-f0-9]{64}$ ]] || exit 2
base_image="python@${base_digest}"
mkdir -p out/image
printf '%s\n' "$base_image" > out/image/base-image.txt
printf '%s\n' "$source_sha" > out/image/source-sha.txt
image="mlsys-worker:$source_sha"
docker build --platform linux/amd64 --build-arg "PYTHON_BASE=$base_image" \
  --build-arg "SOURCE_SHA=$source_sha" -f worker/Dockerfile -t "$image" .
[[ "$(docker image inspect "$image" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" == "$source_sha" ]]
[[ "$(docker run --rm --entrypoint cat "$image" /opt/mlsys/SOURCE_SHA)" == "$source_sha" ]]
docker run --rm "$image" --help
docker run --rm --entrypoint python "$image" -m mlsys_lab.transport --help
set +e
docker run --rm --entrypoint python "$image" -m mlsys_lab.cli smoke \
  --source-sha "$source_sha" --output /tmp/cpu-smoke > out/image/cpu-refusal.json
probe_status=$?
set -e
[[ "$probe_status" == 2 ]] || exit 2
python3 -c 'import json; from pathlib import Path; d=json.loads(Path("out/image/cpu-refusal.json").read_text()); assert d["kernel_verified"] is False'
docker image inspect "$image" --format '{{.Id}}' > out/image/local-image-id.txt
