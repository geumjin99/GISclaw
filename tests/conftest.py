"""Shared fixtures: a throwaway workspace, a stub model, and a test client.

The stub engine answers the ReAct loop with scripted replies, so the whole
server — projects, upload, run, stream, cancel, Toolbox — is exercised without
a network or a key. Set GISCLAW_TEST_SLEEP to make the scripted run linger,
which the cancel and busy tests rely on.
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_WS = tempfile.mkdtemp(prefix="gisclaw-test-")
os.environ["GISCLAW_WORKSPACE"] = _WS
os.environ["DEEPSEEK_API_KEY"] = "test-key-not-real"

from app import runner, server  # noqa: E402  (import after env is set)


class StubEngine:
    """Scripted replies. The `plan` is a list of ReAct replies, consumed in order."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = 0
        self.supports_segmented_user = False

    def load_model(self):
        return True

    def generate(self, prompt="", system_prompt=None, user_message=None, **kw):
        self.calls += 1
        sp = system_prompt or ""
        if "triage incoming messages" in sp:
            return {"text": '{"mode": "analysis"}', "tokens_generated": 5, "latency_ms": 1}
        if "running log" in sp:
            return {"text": "**Result:** stub digest.", "tokens_generated": 5, "latency_ms": 1}
        if "closing note" in sp:
            return {"text": "Stub closing note.", "tokens_generated": 5, "latency_ms": 1}
        text = self.plan.pop(0) if self.plan else \
            'Thought: done\nAction: finish\nArgs: {"summary": "Finished."}'
        return {"text": text, "tokens_generated": 10, "latency_ms": 1}

    def get_stats(self):
        return {"total_calls": self.calls, "total_input_tokens": 100 * self.calls,
                "total_output_tokens": 10 * self.calls, "estimated_cost_usd": 0.0}


PLAN_OK = [
    "Thought: look\nAction: list_files\nArgs: {}",
    'Thought: load\nAction: load_vector\nArgs: {"path": "dataset/pts.geojson", "var_name": "pts"}',
    'Thought: save\nAction: execute\nArgs: {"code": "pts.to_file(\'pred_results/out.geojson\', driver=\'GeoJSON\')\\nprint(len(pts))"}',
    'Thought: done\nAction: finish\nArgs: {"summary": "Two points saved."}',
]

PLAN_SLOW = [
    "Thought: look\nAction: list_files\nArgs: {}",
    'Thought: wait\nAction: execute\nArgs: {"code": "import time\\nfor _ in range(600): time.sleep(0.1)"}',
    'Thought: done\nAction: finish\nArgs: {"summary": "Should not get here."}',
]


@pytest.fixture(scope="session")
def client():
    """A real server on a free loopback port: streaming responses need one.

    Starlette's TestClient collects a whole response before returning it, so
    an SSE run could not be observed while it is happening.
    """
    import socket
    import threading
    import httpx
    import uvicorn

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started, "test server did not start"
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120,
                      headers={"X-GISclaw": "1"}) as c:
        yield c
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def stub(monkeypatch):
    """Install a stub engine; tests pick the plan with stub.use(plan)."""
    holder = {"engine": None}

    def use(plan):
        holder["engine"] = StubEngine(plan)
        return holder["engine"]

    monkeypatch.setattr(runner, "init_llm", lambda cfg: holder["engine"])
    use(PLAN_OK)
    holder["use"] = use
    return holder


GEOJSON = ('{"type":"FeatureCollection","features":['
           '{"type":"Feature","properties":{"n":1},"geometry":{"type":"Point","coordinates":[126.9,37.5]}},'
           '{"type":"Feature","properties":{"n":2},"geometry":{"type":"Point","coordinates":[127.0,37.6]}}]}')


def sse_events(resp):
    """Parse a streamed SSE body into (event, data) pairs."""
    import json
    out = []
    ev, data = "message", ""
    for raw in resp.iter_lines():
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line == "":
            if data:
                try:
                    out.append((ev, json.loads(data)))
                except json.JSONDecodeError:
                    out.append((ev, data))
            ev, data = "message", ""
            continue
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            data += line[5:].strip()
    return out


# ---------------------------------------------------------------- fake Ollama --
CONTEXT_LOADED = 4096


class FakeOllama(BaseHTTPRequestHandler):
    plan = []

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/version":
            return self._json({"version": "0.11.0"})
        if self.path == "/api/tags":
            return self._json({"models": [
                {"name": "qwen2.5-coder:14b", "size": 8_988_000_000,
                 "details": {"parameter_size": "14.8B", "quantization_level": "Q4_K_M", "family": "qwen2"}},
                {"name": "tiny:latest", "size": 400_000_000,
                 "details": {"parameter_size": "0.5B", "quantization_level": "Q8_0", "family": "qwen2"}},
            ]})
        if self.path == "/api/ps":
            return self._json({"models": [{"name": "qwen2.5-coder:14b", "context_length": CONTEXT_LOADED}]})
        if self.path == "/v1/models":
            return self._json({"object": "list", "data": [{"id": "qwen2.5-coder:14b"}, {"id": "tiny:latest"}]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/show":
            return self._json({"parameters": "num_ctx 32768\nstop \"<|im_end|>\"",
                               "model_info": {"qwen2.context_length": 32768}})
        if self.path == "/v1/chat/completions":
            system = next((m["content"] for m in body.get("messages", []) if m["role"] == "system"), "")
            if "triage incoming messages" in system:
                text = '{"mode": "analysis"}'
            elif "running log" in system:
                text = "**Result:** fake digest."
            elif "Reply with the single word" in json.dumps(body):
                text = "ok"
            else:
                text = FakeOllama.plan.pop(0) if FakeOllama.plan else \
                    'Thought: done\nAction: finish\nArgs: {"summary": "Finished."}'
            return self._json({
                "id": "x", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            })
        self._json({"error": "not found"}, 404)


@pytest.fixture(scope="session")
def ollama():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = HTTPServer(("127.0.0.1", port), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1"
    srv.shutdown()


