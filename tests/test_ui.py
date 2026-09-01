"""The front-end in headless Chrome, against the stub-model server.

Skipped when Chrome or selenium is not available. Covers what the API tests
cannot: that Stop stops, that a reloaded page rejoins the run, that the
answer is rendered once, and that no JavaScript error is thrown along the way.
"""
import time

import pytest

from conftest import PLAN_OK, PLAN_SLOW, GEOJSON, FakeOllama

selenium = pytest.importorskip("selenium")
from selenium import webdriver                      # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402
from selenium.webdriver.common.by import By        # noqa: E402


@pytest.fixture(scope="module")
def browser():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1500,950")
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    try:
        d = webdriver.Chrome(options=opts)
    except Exception as e:                          # no Chrome on this machine
        pytest.skip(f"Chrome not available: {e}")
    yield d
    d.quit()


def _wait(cond, secs=25, what=""):
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            if cond():
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise AssertionError(f"timeout waiting for {what}")


def test_run_stop_rejoin_replay(client, stub, browser):
    d = browser
    text = lambda: d.execute_script("return document.getElementById('chatScroll').innerText")
    status = lambda: d.find_element(By.ID, "statusText").text
    running = lambda: "running" in d.find_element(By.CSS_SELECTOR, ".status").get_attribute("class")
    legend_rows = lambda: len(d.find_elements(By.CSS_SELECTOR, "#legendLayers .legend-layer.interactive"))
    open_project = lambda: d.execute_script(
        "[...document.querySelectorAll('#catalog .tree-group')].find(g => g.innerText.includes('UI Check')).click()")
    click_file = lambda name: d.execute_script(
        "[...document.querySelectorAll('#catalog .tree-item')].find(i => i.querySelector('.tree-label').textContent === arguments[0]).click()", name)

    pid = client.post("/api/projects", json={"name": "UI Check"}).json()["id"]
    client.post(f"/api/projects/{pid}/upload?name=pts.geojson", content=GEOJSON.encode())

    d.get(str(client.base_url) + "/")
    _wait(lambda: "UI Check" in d.find_element(By.ID, "catalog").text, what="catalog")
    assert "app.js?v=" in d.page_source
    open_project()
    _wait(lambda: "pts.geojson" in d.find_element(By.ID, "catalog").text, what="tree")

    # 1) a run, stopped from the interface: the sleeping snippet is interrupted
    stub["use"](PLAN_SLOW)
    d.find_element(By.ID, "promptInput").send_keys("loop forever")
    d.find_element(By.ID, "startBtn").click()
    _wait(running, what="running")
    _wait(lambda: "list_files" in text(), what="first step")
    time.sleep(0.8)
    d.find_element(By.ID, "stopBtn").click()
    _wait(lambda: status() == "Stopped", what="stopped")
    assert "Stopped by request" in text()

    # 2) reload mid-run: the page rejoins the run that is still going
    stub["use"](PLAN_SLOW)
    d.find_element(By.ID, "promptInput").send_keys("loop again")
    d.find_element(By.ID, "startBtn").click()
    # the first bubble is folded (its text hidden); a second, open one appears
    _wait(lambda: len(d.find_elements(By.CSS_SELECTOR, ".trace")) >= 2 and "list_files" in text(),
          what="second run started")
    d.refresh()
    _wait(lambda: "UI Check" in d.find_element(By.ID, "catalog").text, what="catalog after reload")
    open_project()
    _wait(lambda: "Rejoined the run" in text(), what="rejoin")
    _wait(running, what="running after rejoin")
    d.find_element(By.ID, "stopBtn").click()
    _wait(lambda: status() == "Stopped", what="stopped after rejoin")

    # 3) a run to completion: the answer once, its layer on the map, data/ and
    #    outputs/ files with the same name kept apart
    stub["use"](list(PLAN_OK))
    d.find_element(By.ID, "promptInput").send_keys("save the points")
    d.find_element(By.ID, "startBtn").click()
    _wait(lambda: status() == "Done", what="done")
    assert text().count("Two points saved.") == 1
    _wait(lambda: "out.geojson" in d.find_element(By.ID, "catalog").text, what="outputs in tree")
    _wait(lambda: legend_rows() == 1, what="result layer on map")
    click_file("pts.geojson")
    _wait(lambda: legend_rows() == 2, what="two layers")

    # 4) history survives a reload; replaying a run shows its answer once more, not twice
    d.refresh()
    _wait(lambda: "UI Check" in d.find_element(By.ID, "catalog").text, what="catalog")
    open_project()
    _wait(lambda: "earlier" in text().lower(), what="history")
    d.find_elements(By.CSS_SELECTOR, ".run-chip.ok")[-1].click()
    _wait(lambda: "end of replay" in text().lower(), what="replay")
    assert text().count("Two points saved.") == 2

    errors = [e["message"] for e in d.get_log("browser")
              if e["level"] == "SEVERE" and "favicon" not in e["message"]]
    assert errors == [], errors


def test_local_models_pane(client, browser, ollama):
    d = browser
    d.get(str(client.base_url) + "/")
    _wait(lambda: d.find_element(By.ID, "catalog"), what="page")
    # Settings → Local models
    d.execute_script("document.querySelector('[data-menu=settings] .menu-btn').click()")
    d.execute_script("document.querySelector('[data-act=set-local]').click()")
    _wait(lambda: not d.find_element(By.ID, "paneLocal").get_attribute("class").count("hidden"), what="local pane")
    url = d.find_element(By.ID, "localUrl")
    url.clear(); url.send_keys(ollama)
    d.find_element(By.ID, "localConnect").click()
    _wait(lambda: "Connected to Ollama" in d.find_element(By.ID, "localStatus").text, what="connect")
    rows = d.find_elements(By.CSS_SELECTOR, "#localModels .lm-row")
    assert len(rows) == 2
    assert "Q4_K_M" in rows[0].text and "9 GB" in rows[0].text and "4,096" in rows[0].text
    # the loaded model has too small a window: the advice is shown
    _wait(lambda: "OLLAMA_CONTEXT_LENGTH" in d.find_element(By.ID, "localAdvice").text, what="advice")
    # add the second model and test it
    d.execute_script("[...document.querySelectorAll('#localModels .lm-add')].pop().click()")
    _wait(lambda: "Added tiny:latest" in d.find_element(By.ID, "localStatus").text, what="added")
    FakeOllama.plan = []
    d.execute_script("[...document.querySelectorAll('#localModels .lm-test')].pop().click()")
    _wait(lambda: "Works" in d.find_element(By.ID, "localStatus").text, what="test")
    # the run selector now offers it
    opts = [o.text for o in d.find_elements(By.CSS_SELECTOR, "#modelSelect option")]
    assert "tiny:latest" in opts
    errors = [e["message"] for e in d.get_log("browser")
              if e["level"] == "SEVERE" and "favicon" not in e["message"]]
    assert errors == [], errors
