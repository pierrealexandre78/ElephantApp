import pytest

from src.model import create_preproc_pipe, create_model_pipe, save_metrics

import pandas as pd

def test_create_preproc_pipeline():
    preproc = create_preproc_pipe()
    assert preproc is not None