"""Headless smoke test of the Streamlit demo (needs `python scripts/train.py` to have produced models/)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "models" / "bundle.joblib").exists(), reason="run scripts/train.py first")
def test_streamlit_app_renders_and_reacts_to_inputs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_demo.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) == 4 and "µs" in at.metric[0].value
    at.sidebar.selectbox[0].select("SBIN").run()                  # change stock => replays another symbol
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.slider[0].set_value(0.8).run()                     # raise the confidence gate
    assert not at.exception, [e.value for e in at.exception]
