"""Look up the uploaded artifact in its owning workflow run."""

import json
import os
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, build_opener

from runpod_ci_broker.provider import NoRedirect, bounded_body

from .budget import REPOSITORY


def uploaded_artifact(run, name, identifier, *, fetch, sleep=time.sleep):
    if not re.fullmatch(r"[1-9][0-9]*", run) or not re.fullmatch(r"[1-9][0-9]*", identifier):
        raise ValueError("invalid_receipt_identity")
    url = f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{run}/artifacts?" + urlencode(
        {"name": name}
    )
    for delay in (1, 2, 4, 8, 0):
        try:
            entries = fetch(url)["artifacts"]
        except HTTPError as exc:
            exc.close()
            if exc.code != 404:
                raise ValueError("artifact_query_refused") from None
            entries = []
        matches = [entry for entry in entries if entry["id"] == int(identifier)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError("artifact_id_ambiguous")
        if delay:
            sleep(delay)
    raise ValueError("artifact_receipt_unavailable")


def github_json(url):
    request = Request(
        url,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
        },
    )
    deadline = time.monotonic() + 4
    with build_opener(NoRedirect).open(request, timeout=4) as response:
        return json.loads(bounded_body(response, 65536, deadline))
