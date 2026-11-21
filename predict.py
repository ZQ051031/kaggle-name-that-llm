import argparse
from pathlib import Path

import joblib
import pandas as pd

from nthllm.modeling import predict
from nthllm.submission import write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('submission.parquet'))
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    test = pd.read_csv(args.data_dir / 'test.csv')
    if test.text.isna().any():
        raise ValueError('Null test text')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(test, predict(joblib.load(args.model), test.text), args.out, args.csv)
    print(f'Validated {len(test)} predictions: {args.out}')


if __name__ == '__main__':
    main()
