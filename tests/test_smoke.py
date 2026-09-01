"""End-to-end smoke test of the server with a stub model. Run: pytest -q"""
import json
import os
import threading
import time

from conftest import GEOJSON, PLAN_OK, PLAN_SLOW, sse_events


def _mkproject(client, name):
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    r = client.post(f"/api/projects/{pid}/upload?name=pts.geojson", content=GEOJSON.encode())
    assert r.status_code == 200, r.text
    return pid


def test_version_and_models(client):
    assert client.get("/api/version").json()["version"]
    models = client.get("/api/models").json()
    assert any(m["id"] == "deepseek" for m in models), models


def test_project_and_files(client):
    pid = _mkproject(client, "Smoke Project")
    tree = client.get(f"/api/projects/{pid}/tree").json()
    assert tree["data"] == ["pts.geojson"]
    assert client.get(f"/api/projects/{pid}/data_check").json()["notices"] == []
    # traversal is refused
    r = client.get(f"/api/projects/{pid}/file", params={"path": "../project.json"})
    assert r.status_code == 400
    r = client.get(f"/api/projects/{pid}/file", params={"path": "pts.geojson"})
    assert r.status_code == 200 and r.json()["type"] == "FeatureCollection"


def test_run_to_completion(client, stub):
    pid = _mkproject(client, "Run OK")
    with client.stream("POST", "/api/run", json={"project_id": pid, "model": "deepseek",
                                                  "instruction": "save the points"}) as r:
        assert r.status_code == 200
        events = sse_events(r)
    kinds = [e for e, _ in events]
    assert kinds[0] == "run"
    assert "step" in kinds and "result" in kinds and "summary" in kinds
    done = [d for e, d in events if e == "done"][0]
    assert done["success"] is True and done["output_files"] == ["out.geojson"]
    summary = [d for e, d in events if e == "summary"][0]
    assert summary["content"] == "Two points saved."
    # durable record
    entries = client.get(f"/api/projects/{pid}/chat").json()["entries"]
    assert entries[-1]["role"] == "agent" and entries[-1]["outputs"] == ["out.geojson"]
    assert entries[-1]["final_summary"] == "Two points saved."
    assert "out.geojson" in client.get(f"/api/projects/{pid}/tree").json()["outputs"]
    assert "Smoke" not in client.get(f"/api/projects/{pid}/journal").json()["markdown"]
    # the run folder holds the audit trail
    run_id = done["run_id"]
    trace = client.get(f"/api/projects/{pid}/trace", params={"run": run_id}).json()
    assert len(trace["events"]) == 4 and "to_file" in trace["code"]
    assert client.get("/api/run/active").json()["active"] is None


def test_cancel_busy_and_rejoin(client, stub):
    stub["use"](PLAN_SLOW)
    pid = _mkproject(client, "Run Slow")
    got = {}

    def consume():
        with client.stream("POST", "/api/run", json={"project_id": pid, "model": "deepseek",
                                                      "instruction": "loop"}) as r:
            got["status"] = r.status_code
            got["events"] = sse_events(r)

    t = threading.Thread(target=consume)
    t.start()
    # wait until the slow snippet has started executing
    active = None
    for _ in range(100):
        active = client.get("/api/run/active", params={"project": pid}).json()["active"]
        if active and any(d.get("action") == "list_files" for d in runner_events(active["run_id"])):
            break
        time.sleep(0.1)
    time.sleep(0.5)
    assert active and not active["done"]
    run_id = active["run_id"]

    # a second run and a Toolbox operation are refused while it runs
    r = client.post("/api/run", json={"project_id": pid, "model": "deepseek", "instruction": "x"})
    assert r.status_code == 409 and r.json()["busy"]
    r = client.post(f"/api/projects/{pid}/geoprocess",
                    json={"op": "centroid", "inputs": {"layer": "pts.geojson"}, "output": "c"})
    assert r.status_code == 409

    # rejoining replays what was emitted so far
    with client.stream("GET", f"/api/run/{run_id}/stream") as r2:
        assert r2.status_code == 200
        # stop while both streams are open; the running snippet is interrupted
        t0 = time.time()
        assert client.post(f"/api/run/{run_id}/cancel").json()["ok"] is True
        replay = sse_events(r2)
    t.join(timeout=30)
    assert not t.is_alive()
    took = time.time() - t0
    assert took < 15, f"stop took {took:.1f}s"
    done = [d for e, d in got["events"] if e == "done"][0]
    assert done["stopped"] is True and done["success"] is False
    assert [e for e, _ in replay][0] == "run"
    assert [d for e, d in replay if e == "done"][0]["stopped"] is True
    entries = client.get(f"/api/projects/{pid}/chat").json()["entries"]
    assert entries[-1].get("stopped") is True
    assert client.get("/api/run/active").json()["active"] is None


def runner_events(run_id):
    from app import runner
    r = runner.get_run(run_id)
    return list(r.events) if r else []


def test_toolbox_is_recorded(client):
    pid = _mkproject(client, "Toolbox")
    r = client.post(f"/api/projects/{pid}/geoprocess",
                    json={"op": "buffer", "inputs": {"layer": "pts.geojson"},
                          "params": {"distance": 100}, "output": "buf"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["outputs"][0]["filename"] == "buf.geojson"
    entries = client.get(f"/api/projects/{pid}/chat").json()["entries"]
    assert entries[-1]["kind"] == "tool" and entries[-1]["outputs"] == ["buf.geojson"]
    assert "Toolbox: buffer" in client.get(f"/api/projects/{pid}/journal").json()["markdown"]
    code = os.path.join(os.environ["GISCLAW_WORKSPACE"], pid, "runs", body["run_id"], "code.py")
    assert os.path.isfile(code) and "buffer" in open(code).read()


def test_settings_mask_keys(client):
    r = client.post("/api/settings/providers/openai", json={"api_key": "sk-abcdefghijklmnop1234"})
    assert r.status_code == 200
    provs = {p["id"]: p for p in client.get("/api/settings").json()["providers"]}
    assert provs["openai"]["masked_key"] == "sk-ab…1234"
    assert "abcdefghijkl" not in json.dumps(provs)
    st = os.stat(os.path.join(os.environ["GISCLAW_WORKSPACE"], ".gisclaw", "settings.json"))
    assert oct(st.st_mode & 0o777) == "0o600"


def test_cross_site_requests_are_refused(client):
    import httpx
    base = str(client.base_url)
    with httpx.Client(base_url=base) as bare:          # no client header
        r = bare.post("/api/projects", json={"name": "x"})
        assert r.status_code == 403
        r = bare.get("/api/projects")                    # reads are fine
        assert r.status_code == 200
    r = client.post("/api/projects", json={"name": "x"},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/projects", json={"name": "Same Origin"},
                    headers={"Origin": base.rstrip("/")})
    assert r.status_code == 200


def test_interface_language_setting(client):
    assert client.get("/api/settings").json()["language"] in ("", "en", "zh", "ko")
    assert client.post("/api/settings/ui", json={"language": "ko"}).json()["language"] == "ko"
    assert client.get("/api/settings").json()["language"] == "ko"
    assert client.post("/api/settings/ui", json={"language": "xx"}).json()["language"] == ""
    client.post("/api/settings/ui", json={"language": "en"})
