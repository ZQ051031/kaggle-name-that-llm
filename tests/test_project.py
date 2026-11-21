import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from nthllm import LABELS
from nthllm.features import LocalPerplexity, StyleFeatures
from nthllm.submission import validate_submission
from scripts.make_synthetic import generate


ROOT = Path(__file__).resolve().parents[1]


def test_features():
    extractor = StyleFeatures()
    texts = ['', 'The cat — “hello”; **bold**\n- one\n1. Two… (3).', 'We run. However, we wait!']
    X = extractor.fit_transform(texts)
    assert X.shape == (3, len(extractor.get_feature_names_out()))
    assert np.isfinite(X).all()
    names = list(extractor.get_feature_names_out())
    assert X[1, names.index('em_dash')] > 0
    assert X[1, names.index('curly_quotes')] > 0
    assert X[2, names.index('starts_however')] > 0
    assert extractor.transform([]).shape[0] == 0


def test_optional_lm(tmp_path):
    assert not LocalPerplexity().transform(['', 'hello']).any()
    with pytest.warns(RuntimeWarning, match='unavailable'):
        assert not LocalPerplexity(str(tmp_path / 'absent')).transform(['hello']).any()


@pytest.mark.parametrize('issue', ['columns', 'duplicate', 'null', 'label', 'missing', 'extra'])
def test_submission_errors(issue):
    test = pd.DataFrame({'id': [1, 2]})
    sub = pd.DataFrame({'id': [1, 2], 'model': [LABELS[0], LABELS[1]]})
    if issue == 'columns':
        sub = sub[['model', 'id']]
    elif issue == 'duplicate':
        sub.loc[1, 'id'] = 1
    elif issue == 'null':
        sub.loc[1, 'model'] = None
    elif issue == 'label':
        sub.loc[1, 'model'] = 'unknown'
    elif issue == 'missing':
        sub = sub.iloc[:1]
    else:
        sub.loc[1, 'id'] = 3
    with pytest.raises(ValueError):
        validate_submission(sub, test)


def test_empty_submission():
    empty = pd.DataFrame({'id': pd.Series(dtype=int), 'model': pd.Series(dtype=str)})
    assert validate_submission(empty, empty[['id']]).empty


def test_end_to_end(tmp_path):
    train, test = generate(tmp_path, prompts=12)
    assert set(train.model) == set(LABELS)
    assert train.text.str.split().str.len().between(40, 250).all()
    assert train.groupby('model').size().nunique() > 1
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    def run(*args):
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    model = tmp_path / 'model.joblib'
    run('train.py', '--data-dir', tmp_path, '--out', model, '--folds', 2)
    report = json.loads(model.with_suffix('.cv.json').read_text())
    for split in report['splits']:
        assert not set(split['train_prompts']) & set(split['valid_prompts'])
    assert set(report['models']) == {'logistic', 'calibrated_svc', 'sgd', 'ensemble'}
    run('predict.py', '--data-dir', tmp_path, '--model', model, '--out', tmp_path / 'sub.parquet')
    validate_submission(pd.read_parquet(tmp_path / 'sub.parquet'), test)
