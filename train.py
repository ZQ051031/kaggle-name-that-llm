import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nthllm.modeling import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--lm-path', help='Optional local causal-LM directory; never downloads')
    args = parser.parse_args()
    artifact, report = train(pd.read_csv(args.data_dir / 'train.csv'), args.folds, args.lm_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    report_path = args.out.with_suffix('.cv.json')
    report_path.write_text(json.dumps(report, indent=2))
    for name, result in report['models'].items():
        print(f"{name}: {result['mean']:.4f} ± {result['std']:.4f}")
        print('per-class recall:', result['per_class_recall'])
    print(f"Runtime: {report['runtime_seconds']:.2f}s; report: {report_path}")


if __name__ == '__main__':
    main()
