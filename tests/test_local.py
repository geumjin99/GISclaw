"""A model server on this machine, end to end, against a fake Ollama.

The fake speaks Ollama's own API (version, tags, show, ps) and the
OpenAI-compatible chat endpoint, so the probe, the add, the Test button and a
full run all go over the wire through the real engine — nothing is stubbed
inside the server.
"""
from conftest import CONTEXT_LOADED, GEOJSON, PLAN_OK, FakeOllama, sse_events

def test_local_probe_add_test_run(client, ollama):
    # the pane's starting point
    info = client.get("/api/settings/local").json()
    assert "ollama" in info["presets"] and info["min_context"] == 8192

    # probe: Ollama recognised, models with their sizes and context limits
    res = client.get("/api/settings/local/probe", params={"base_url": ollama}).json()
    assert res["ok"] and res["kind"] == "ollama" and res["version"] == "0.11.0"
    m = {x["id"]: x for x in res["models"]}
    assert m["qwen2.5-coder:14b"]["size_gb"] == 9.0
    assert m["qwen2.5-coder:14b"]["context_max"] == 32768
    assert m["qwen2.5-coder:14b"]["context_set"] == 32768
    assert m["qwen2.5-coder:14b"]["context_chars"] > 24_000
    assert res["running"][0]["context"] == CONTEXT_LOADED
    # the address was saved as the provider endpoint
    provs = {p["id"]: p for p in client.get("/api/settings").json()["providers"]}
    assert provs["local"]["base_url"] == ollama and provs["local"]["configured"]

    # add it the way the pane does
    r = client.post("/api/settings/models", json={
        "id": "qwen2.5-coder:14b", "display": "qwen2.5-coder:14b", "provider": "local",
        "model_name": "qwen2.5-coder:14b", "max_rounds": 35, "max_tokens": 2048,
        "timeout": 600, "context_chars": m["qwen2.5-coder:14b"]["context_chars"]})
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    mine = {x["id"]: x for x in r.json()["models"]}[mid]
    assert mine["ready"] and mine["provider"] == "local" and mine["timeout"] == 600
    assert mine["context_chars"] == m["qwen2.5-coder:14b"]["context_chars"]

    # Test: a real call through the engine, then the loaded context is reported
    t = client.post("/api/settings/providers/local/test", json={"model_name": "qwen2.5-coder:14b"}).json()
    assert t["ok"] and t["reply"] == "ok"
    assert t["context_length"] == CONTEXT_LOADED
    assert "OLLAMA_CONTEXT_LENGTH" in t["context_advice"]

    # a full run over the wire with the real OpenAI-compatible engine
    FakeOllama.plan = list(PLAN_OK)
    pid = client.post("/api/projects", json={"name": "Local Run"}).json()["id"]
    client.post(f"/api/projects/{pid}/upload?name=pts.geojson", content=GEOJSON.encode())
    with client.stream("POST", "/api/run", json={"project_id": pid, "model": mid,
                                                  "instruction": "save the points"}) as r:
        assert r.status_code == 200
        events = sse_events(r)
    done = [d for e, d in events if e == "done"][0]
    assert done["success"] is True and done["output_files"] == ["out.geojson"]
    assert done["cost"]["api_calls"] >= 5 and done["cost"]["cost_usd"] == 0


def test_local_probe_unreachable(client):
    res = client.get("/api/settings/local/probe", params={"base_url": "http://127.0.0.1:9/v1"}).json()
    assert res["ok"] is False and "Nothing answered" in res["error"]
