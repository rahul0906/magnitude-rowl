#!/usr/bin/env python3
"""Run one bounded Gemma request through the installed Magnitude service."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


MAGNITUDE = Path.home() / ".magnitude/bin/magnitude"
MODEL_MARKERS = (
    "hauhaucs/gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced",
    "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced",
)
PORT = 10100
PHASE = "starting"


def get_json(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def run_cli(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(MAGNITUDE), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "TERM": "dumb"},
    )


def candidate_models(value: object) -> list[dict]:
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(candidate_models(item))
        return result
    if not isinstance(value, dict):
        return []

    text = json.dumps(value, ensure_ascii=False).casefold()
    has_id = any(isinstance(value.get(key), str) for key in ("id", "modelId", "canonicalId"))
    if has_id and any(marker in text for marker in MODEL_MARKERS) and "q4_k_m" in text:
        return [value]
    result = []
    for item in value.values():
        result.extend(candidate_models(item))
    return result


def model_id(model: dict) -> str | None:
    for key in ("id", "modelId", "canonicalId"):
        value = model.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def stop_group(process: subprocess.Popen[str]) -> None:
    pid = process.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        # An in-place app update can replace the service process during a
        # request and leave the original process-group handle non-signalable.
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=5)


def main() -> int:
    global PHASE
    run_dir = Path(os.environ["MRIG_RUN"])
    log_path = run_dir / "installed-magnitude-gemma.log"
    models_path = run_dir / "installed-magnitude-models.json"
    load_path = run_dir / "installed-magnitude-load.json"

    PHASE = "checking service status"
    status = run_cli("status", timeout=10).stdout.casefold()
    if "not running" not in status:
        raise RuntimeError("Magnitude is already serving; refusing to take over its service")

    with log_path.open("w", encoding="utf-8") as log:
        service = subprocess.Popen(
            [str(MAGNITUDE), "serve"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "TERM": "dumb"},
        )
        selected: str | None = None
        restore_drafter: str | None = None
        try:
            PHASE = "waiting for service API"
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if service.poll() is not None:
                    raise RuntimeError(f"Magnitude service exited early; see {log_path}")
                try:
                    get_json(f"http://127.0.0.1:{PORT}/inference/v1/models")
                    break
                except (OSError, urllib.error.URLError, json.JSONDecodeError):
                    time.sleep(1)
            else:
                raise RuntimeError(f"Magnitude service did not become ready; see {log_path}")

            PHASE = "reading installed model inventory"
            inventory = json.loads(run_cli("models", "list", "--all", "--json").stdout)
            models_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
            matches = candidate_models(inventory)
            ids = {model_id(item) for item in matches}
            ids.discard(None)
            if len(ids) != 1:
                raise RuntimeError(
                    f"expected exactly one installed Gemma Q4_K_M model; found ids {sorted(ids)}; "
                    f"inventory saved to {models_path}"
                )
            selected = ids.pop()
            print(f"runtime: Magnitude {run_cli('--version', timeout=10).stdout.strip()}")
            print(f"model: {selected}")
            print(f"inventory: {models_path}")

            selected_record = next(item for item in matches if model_id(item) == selected)
            choice = selected_record.get("drafterChoice", {})
            choice_tag = choice.get("_tag") if isinstance(choice, dict) else None
            if choice_tag != "Auto":
                raise RuntimeError(
                    f"expected the installed model's drafter choice to be Auto before the control; found {choice_tag!r}"
                )
            PHASE = "temporarily disabling the selected drafter"
            run_cli("models", "drafter", selected, "none", "--json")
            restore_drafter = "auto"
            print("drafter: temporarily disabled for the explicit control")

            PHASE = "requesting model load through the Magnitude CLI"
            load = run_cli("models", "load", selected, "--json", timeout=60)
            load_path.write_text(load.stdout, encoding="utf-8")
            run_id = run_dir.parent.name
            signal_path = Path("/private/tmp/codex-ssd-gemma-ui") / f"{run_id}.done"
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.unlink(missing_ok=True)
            print(f"ui: http://127.0.0.1:{PORT}")
            print(f"model: {selected}")
            print("prompt: What is 2+2? Reply with one digit.")
            print(f"complete the chat, then create this signal file: {signal_path}")
            print(f"model-load response: {load_path}")

            PHASE = "waiting for the local UI request to finish"
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if signal_path.is_file():
                    print(f"ui completion signal received: {signal_path}")
                    return 0
                if service.poll() is not None:
                    raise RuntimeError(f"Magnitude service exited while waiting; see {log_path}")
                time.sleep(0.5)
            raise TimeoutError(f"no UI completion signal within 10 minutes; see {log_path}")
        finally:
            if selected is not None and restore_drafter is not None:
                try:
                    run_cli("models", "drafter", selected, restore_drafter, "--json")
                    print("drafter: restored Auto")
                except Exception as restore_error:
                    print(f"failed to restore the original Auto drafter choice: {restore_error}", file=sys.stderr)
            stop_group(service)
            status = run_cli("status", timeout=10).stdout.casefold()
            if "not running" not in status:
                raise RuntimeError("Magnitude service remained active after the smoke")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"installed Magnitude Gemma check failed while {PHASE}: {error}", file=sys.stderr)
        raise SystemExit(1)
