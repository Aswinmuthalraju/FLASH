import copy

import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c = copy.deepcopy(c)
    c["synthetic"]["n_days"] = 12          # small + fast for unit tests
    c["symbols"] = c["symbols"][:4]
    return c


@pytest.fixture(scope="session")
def synth(cfg):
    from src.data.build import build_synthetic

    return build_synthetic(cfg, save=False)
