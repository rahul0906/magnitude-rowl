#!/bin/bash
set -euo pipefail

# A small compatibility check for one pinned Inkling quant. The default is
# IQ1_S; set INKLING_PROFILE=q4k or unsloth-q4kxl for a quality candidate. The expert store
# is kept under the user's cache, and server configuration and logs stay in
# this mrig run.

: "${MRIG_RUN:?mrig must provide MRIG_RUN}"

MOE_PREFIX="$(brew --prefix moeshift)"
MOESHIFT_BIN="$(command -v moeshift)"
MOE_PYTHON="${MOE_PREFIX}/libexec/bin/python"
EXPERT_TOOL="${MOE_PREFIX}/share/moeshift/tools/moe/experts.py"
INKLING_REVISION="309509848b3151419aa26d069eb03db57fd29a00"
INKLING_PROFILE="${INKLING_PROFILE:-iq1s}"
INKLING_CONTEXT="${INKLING_CONTEXT:-4096}"
INKLING_CHECK="${INKLING_CHECK:-short}"
case "${INKLING_PROFILE}" in
  iq1s) INKLING_FILE="Inkling-2-Mini-AD-IQ1_S.gguf"; EXPERT_BYTES=5478285312
        INKLING_REPO="AtomicChat"
        EXPECTED_SHA256="4d3cab7e88016548136e13f7ec465db7fc49ab8cf9559a60f3daad1f3f15c9d5" ;;
  q4k)  INKLING_FILE="Inkling-2-Mini-AD-Q4_K.gguf"; EXPERT_BYTES=14778630144
        INKLING_REPO="AtomicChat"
        EXPECTED_SHA256="2bfc233e10925297cc79dffce5d9be4f2a2b43e499ea6a1bf03b9242867dce9e" ;;
  unsloth-q4kxl) INKLING_FILE="Inkling-2-Mini-UD-Q4_K_XL.gguf"; EXPERT_BYTES=15797846016
        INKLING_REPO="unsloth"
        INKLING_REVISION="8078d149286eec7193795413b9e0643da6d11fa1"
        EXPECTED_SHA256="e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3" ;;
  *) echo "unsupported Inkling profile: ${INKLING_PROFILE}" >&2; exit 2 ;;
esac
if [[ ! "${INKLING_CONTEXT}" =~ ^[1-9][0-9]{0,8}$ ]] || (( INKLING_CONTEXT < 1024 )); then
  echo "INKLING_CONTEXT must be an integer of at least 1024 tokens" >&2
  exit 2
fi
INKLING_GGUF="${HOME}/.cache/huggingface/hub/models--${INKLING_REPO}--Inkling-2-Mini-GGUF/snapshots/${INKLING_REVISION}/${INKLING_FILE}"
EXPERTS_DIR="${HOME}/.cache/magnitude/moeshift/inkling-2-mini-${INKLING_PROFILE}-${INKLING_REVISION}"
CONFIG="${MRIG_RUN}/moeshift-inkling-${INKLING_PROFILE}.json"
SERVER_LOG="${MRIG_RUN}/moeshift-inkling-${INKLING_PROFILE}.log"
RESPONSE="${MRIG_RUN}/moeshift-inkling-${INKLING_PROFILE}-response.json"
PID_FILE="${MRIG_RUN}/moeshift-inkling-${INKLING_PROFILE}.pid"
LAUNCHER_PID=""
SAMPLER_PID=""

cleanup() {
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  if [[ -f "${PID_FILE}" ]]; then
    python3 - "$(cat "${PID_FILE}")" <<'PY'
import os
import signal
import sys
import time

process_group = int(sys.argv[1])
try:
    os.killpg(process_group, signal.SIGTERM)
except ProcessLookupError:
    pass
for _ in range(50):
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        break
    time.sleep(0.1)
else:
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
PY
  fi
  if [[ -n "${LAUNCHER_PID}" ]]; then
    kill "${LAUNCHER_PID}" 2>/dev/null || true
    wait "${LAUNCHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -s "${INKLING_GGUF}" ]]; then
  echo "missing Inkling checkpoint: ${INKLING_GGUF}" >&2
  exit 1
fi
read -r model_sha256 _ < <(shasum -a 256 "${INKLING_GGUF}")
if [[ "${model_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Inkling checkpoint hash does not match the pinned profile: ${model_sha256}" >&2
  exit 1
fi
if [[ ! -x "${MOE_PYTHON}" || ! -f "${EXPERT_TOOL}" ]]; then
  echo "MoeShift expert preparation tools are missing under ${MOE_PREFIX}" >&2
  exit 1
fi

mkdir -p "$(dirname "${EXPERTS_DIR}")"
if [[ -f "${EXPERTS_DIR}/manifest.json" ]]; then
  "${MOE_PYTHON}" "${EXPERT_TOOL}" verify --model "${INKLING_GGUF}" --experts "${EXPERTS_DIR}"
elif [[ -d "${EXPERTS_DIR}" ]] && [[ -n "$(find "${EXPERTS_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite incomplete expert store: ${EXPERTS_DIR}" >&2
  exit 1
else
  free_bytes="$(python3 - "$(dirname "${EXPERTS_DIR}")" <<'PY'
import shutil
import sys

print(shutil.disk_usage(sys.argv[1]).free)
PY
)"
  minimum_free_bytes=$((EXPERT_BYTES + 15 * 1024 * 1024 * 1024))
  if (( free_bytes < minimum_free_bytes )); then
    echo "need at least ${minimum_free_bytes} free bytes before extracting Inkling experts; have ${free_bytes}" >&2
    exit 2
  fi
  "${MOE_PYTHON}" "${EXPERT_TOOL}" extract --model "${INKLING_GGUF}" --out-dir "${EXPERTS_DIR}"
  "${MOE_PYTHON}" "${EXPERT_TOOL}" verify --model "${INKLING_GGUF}" --experts "${EXPERTS_DIR}"
fi

PORT="$(python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    print(sock.getsockname()[1])
PY
)"

python3 - "${CONFIG}" "${INKLING_GGUF}" "${EXPERTS_DIR}" "${INKLING_CONTEXT}" "${PORT}" <<'PY'
import json
import sys
from pathlib import Path

config_path, model, experts, context, port = sys.argv[1:]
profile = {
    "args": {
        "model": model,
        "moe-experts": experts,
        "alias": "inkling-2-mini",
        "ctx-size": int(context),
        "ctx-checkpoints": 1,
        "prefill-max-tokens": 1024,
        "moe-prefill-bank": 128,
        "moe-decode-bank": 0,
        "n-gpu-layers": 99,
        "flash-attn": "on",
        "cache-type-k": "f16",
        "cache-type-v": "f16",
        "threads": 8,
        "no-mmap": True,
    },
    "env": {
        "LLAMA_MOE_QUIET": 0,
        "LLAMA_GPU_KEEPALIVE": 1,
        "LLAMA_MOE_DECODE_WINDOW": 12,
        "LLAMA_MOE_LOOKAHEAD_PREFETCH": 1,
        "LLAMA_MOE_PREFETCH_READ": 2,
        "LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH": 3,
        "LLAMA_MOE_PREFILL_PIPELINE": 1,
        "LLAMA_MOE_PROTECTED_SLOTS": 6,
    },
}
config = {
    "host": "127.0.0.1",
    "port": int(port),
    "timeout": 1800,
    "models": {"inkling-2-mini": profile},
    "check-updates": False,
}
Path(config_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

echo "runtime: $("${MOESHIFT_BIN}" --version)"
echo "model: ${INKLING_GGUF}"
echo "revision: ${INKLING_REVISION}"
echo "model sha256: ${model_sha256}"
echo "experts: ${EXPERTS_DIR}"
echo "context capacity: ${INKLING_CONTEXT}"

python3 - "${PID_FILE}" "${MOESHIFT_BIN}" "${CONFIG}" "${PORT}" <<'PY' >"${SERVER_LOG}" 2>&1 &
import subprocess
import sys
import time
from pathlib import Path

pid_file, binary, config, port = sys.argv[1:]
process = subprocess.Popen(
    [binary, "--config", config, "--port", port],
    start_new_session=True,
    stdout=sys.stdout,
    stderr=subprocess.STDOUT,
)
Path(pid_file).write_text(f"{process.pid}\n", encoding="ascii")
raise SystemExit(process.wait())
PY
LAUNCHER_PID=$!

# `footprint -t` deduplicates shared memory across the server and its
# descendants. Keep the raw samples and system swap alongside the request;
# these sparse samples are not a claim about an unobserved peak.
python3 - "${PID_FILE}" "${MRIG_RUN}/moeshift-inkling-${INKLING_PROFILE}-memory.jsonl" "${INKLING_CHECK}" <<'PY' &
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

pid_path, output_path = map(Path, sys.argv[1:3])
check = sys.argv[3]
for _ in range(100):
    if pid_path.is_file():
        break
    time.sleep(0.1)
else:
    raise SystemExit("server did not publish its process id")
pid = int(pid_path.read_text())
initial_swap_mib = None
with output_path.open("w", encoding="utf-8") as output:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        sample = {"time": time.time(), "pid": pid}
        for name, command in (
            ("footprint", ["footprint", "-t", "-f", "bytes", "--swapped", str(pid)]),
            ("swap", ["sysctl", "vm.swapusage"]),
            ("vm_stat", ["vm_stat"]),
        ):
            try:
                result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
                sample[name] = {"code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            except (OSError, subprocess.TimeoutExpired) as error:
                sample[name] = {"error": str(error)}
        swap = re.search(r"used = ([0-9.]+)([MG])", sample.get("swap", {}).get("stdout", ""))
        if swap:
            used_mib = float(swap.group(1)) * (1024 if swap.group(2) == "G" else 1)
            if initial_swap_mib is None:
                initial_swap_mib = used_mib
            if check == "retrieval-32k" and used_mib - initial_swap_mib > 1024:
                sample["stopped"] = "system swap grew by more than 1 GiB during the long-context check"
        output.write(json.dumps(sample) + "\n")
        output.flush()
        if "stopped" in sample:
            print(sample["stopped"], file=sys.stderr, flush=True)
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            break
        time.sleep(10)
PY
SAMPLER_PID=$!

ready=0
for ((attempt = 0; attempt < 600; attempt++)); do
  if curl --fail --silent --max-time 2 "http://127.0.0.1:${PORT}/v1/models" > "${MRIG_RUN}/moeshift-inkling-${INKLING_PROFILE}-models.json" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
    echo "MoeShift exited before its API became ready:" >&2
    cat "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 1
done
if (( ready == 0 )); then
  echo "MoeShift did not become ready within 10 minutes:" >&2
  cat "${SERVER_LOG}" >&2
  exit 1
fi

python3 - "${PORT}" "${RESPONSE}" "${INKLING_CHECK}" <<'PY'
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

port, response_path, check = sys.argv[1:]
if check == "short":
    prompt = "What is 2+2? Reply with one digit."
    expected = ("4",)
elif check in ("retrieval", "retrieval-32k"):
    cycle = [
        f"// service {i:03d}: worker={i % 9}, timeout_ms={300 + i}, retry_limit={i % 5}, cache_bucket=bucket-{i:03d}.\n"
        for i in range(80)
    ]
    records = cycle if check == "retrieval" else cycle * 12 + cycle[:60]
    middle = len(records) // 2
    prompt = (
        "Read the configuration records and answer the question at the end.\n"
        "ATLAS_RELEASE_COLOR=COBALT-4817\n"
        + "".join(records[:middle])
        + "HARBOR_RELEASE_COLOR=MAPLE-9302\n"
        + "".join(records[middle:])
        + "What are the exact values of ATLAS_RELEASE_COLOR and HARBOR_RELEASE_COLOR? Reply with both values."
    )
    expected = ("COBALT-4817", "MAPLE-9302")
else:
    raise SystemExit(f"unsupported Inkling check: {check}")
messages = [{"role": "user", "content": prompt}]
chat = {
    "model": "inkling-2-mini",
    "messages": messages,
    "chat_template_kwargs": {"reasoning_effort": "none"},
}
Path(response_path).with_name(f"moeshift-inkling-{check}-request.json").write_text(
    json.dumps(chat, indent=2) + "\n", encoding="utf-8"
)
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({
        **chat,
        "temperature": 0,
        "max_tokens": 512 if check != "short" else 64,
        "stream": False,
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=1800) as result:
        payload = json.load(result)
except urllib.error.HTTPError as error:
    raise SystemExit(f"completion request failed: HTTP {error.code}: {error.read().decode('utf-8', 'replace')}") from error
Path(response_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
choices = payload.get("choices", [])
if not choices:
    raise SystemExit(f"completion has no choices: {payload}")
content = choices[0].get("message", {}).get("content", "")
if not isinstance(content, str) or not content.strip():
    raise SystemExit(f"completion has no text: {payload}")
if any(value not in content for value in expected):
    raise SystemExit(f"completion did not include the expected answer: {payload}")
if choices[0].get("finish_reason") != "stop":
    raise SystemExit(f"completion did not stop normally: {payload}")
minimum_tokens = {"retrieval": 1000, "retrieval-32k": 30000}.get(check, 0)
if payload.get("usage", {}).get("prompt_tokens", 0) < minimum_tokens:
    raise SystemExit(f"retrieval prompt was shorter than intended: {payload.get('usage')}")
print(f"completion: {content.strip()}")
print(f"finish_reason: {choices[0].get('finish_reason')}")
print(f"usage: {payload.get('usage')}")
PY

if [[ "${INKLING_CHECK}" == "retrieval-32k" ]] && ! rg --quiet 'truncated = 0' "${SERVER_LOG}"; then
  echo "the server did not confirm an untruncated long-context request" >&2
  exit 1
fi

echo "server log: ${SERVER_LOG}"
