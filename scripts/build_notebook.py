"""Rebuild the standalone offline Kaggle script after changes to nthllm."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build():
    sources = {f'nthllm/{name}.py': (ROOT / 'nthllm' / f'{name}.py').read_text()
               for name in ('__init__', 'features', 'modeling', 'submission')}
    header = '''# Standalone Kaggle notebook script. Internet must remain OFF.
# Public pretrained weights, if used, must be attached as a Kaggle dataset.
import os
from pathlib import Path
import sys
import tempfile

'''
    footer = '''
bundle = Path(tempfile.mkdtemp(prefix='nthllm-offline-'))
for name, source in SOURCES.items():
    target = bundle / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
sys.path.insert(0, str(bundle))

import pandas as pd
from nthllm.modeling import make_features, make_models, predict
from nthllm.submission import write_submission
from sklearn.model_selection import GroupKFold

def main():
    data_dir = Path(os.environ.get('NTHLLM_DATA_DIR', '/kaggle/input/competitions/name-that-llm'))
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    features = make_features(os.environ.get('NTHLLM_LM_PATH'))
    X = features.fit_transform(train.text)
    models = make_models()
    models['calibrated_svc'].cv = list(GroupKFold(3).split(X, train.model, train.prompt_id))
    for model in models.values():
        model.fit(X, train.model)
    artifact = {'features': features, 'models': models}
    out = os.environ.get('NTHLLM_OUT', 'submission.parquet')
    write_submission(test, predict(artifact, test.text), out)
    print(f'Offline notebook: validated {len(test)} rows -> {out}')

if __name__ == '__main__':
    main()
'''
    (ROOT / 'notebooks/kaggle_submission.py').write_text(header + 'SOURCES = ' + repr(sources) + '\n' + footer)


if __name__ == '__main__':
    build()
