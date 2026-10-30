"""The per-bar code must not touch pandas, a database, the network, the filesystem or model loading."""
import inspect

from src.inference.feature_cache import FeatureState, NewsCache
from src.inference.predictor import Predictor
from src.models.calibration import FastCalibrator
from src.models.hmm_regime import ForwardFilter

FORBIDDEN = ["pd.", "pandas", "duckdb", "urllib", "requests", "open(", "joblib", "np.load", "read_csv", "read_parquet",
             "load_booster", "lgb.Booster", "from_pretrained", "torch", "sklearn", ".fit(", "viterbi", "decode("]
HOT = [Predictor.on_bar, FeatureState.update, NewsCache.read, NewsCache.set, ForwardFilter.step, FastCalibrator.__call__]


def test_hot_path_sources_are_clean():
    for fn in HOT:
        src = inspect.getsource(fn)
        for tok in FORBIDDEN:
            assert tok not in src, f"{fn.__qualname__} contains {tok!r}"
