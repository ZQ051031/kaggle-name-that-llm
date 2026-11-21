import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nthllm import LABELS


TOPICS = {
    'philosophy': 'ethics reason belief knowledge meaning freedom truth mind',
    'physics': 'energy motion particle field force matter wave gravity',
    'math': 'proof number theorem function sequence geometry equation symmetry',
    'biology': 'cell organism gene evolution protein tissue species ecology',
    'CS': 'algorithm memory network program data compiler system complexity',
    'creative prose': 'river shadow lantern garden traveler window rain dawn',
}
OPENERS = ['However', 'In practice', 'Consider this', 'Broadly speaking', 'Notably', 'For example']
CONNECTORS = ['and', 'yet', 'because', 'while', 'although', 'so']


def passage(rng, label_index, prompt):
    topic = list(TOPICS.values())[prompt % len(TOPICS)].split()
    # Mixture noise deliberately prevents any one surface marker identifying a class.
    style = label_index if rng.random() < 0.78 else int(rng.integers(18))
    target = int(rng.integers(65, 180))
    sentences = []
    while len(' '.join(sentences).split()) < target:
        opener = OPENERS[style % 6] if rng.random() < 0.55 else rng.choice(OPENERS)
        connector = CONNECTORS[style // 3] if rng.random() < 0.55 else rng.choice(CONNECTORS)
        a, b, d = rng.choice(topic, 3, replace=False)
        phrase = rng.choice([
            f'{opener}, the {a} reveals how {b} can change over time {connector} we examine the {d}',
            f'{opener}, we should compare {a} with {b} {connector} this approach offers a useful account of {d}',
            f'{opener}, a careful view of {a} suggests that {b} is not simply explained by {d}',
            f'{opener}, imagine a different {a} where the {b} becomes a question rather than an answer about {d}',
        ])
        if rng.random() < 0.35:
            phrase += rng.choice([' in a broader context', ' with some uncertainty', ' rather than a fixed rule'])
        punctuation = ['; ', ' — ', ': '][style // 6] if rng.random() < 0.55 else ', '
        phrase = phrase.replace(', ', punctuation, 1)
        if rng.random() < 0.2:
            phrase = phrase.replace('careful', '“careful”').replace('useful', '"useful"')
        sentences.append(phrase + '.')
    separator = '\n' if rng.random() < (0.12 + (style % 3)*0.2) else ' '
    return separator.join(sentences)


def generate(out, prompts=150, seed=42):
    if prompts < 6:
        raise ValueError('Use at least six prompts')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    # Every class occurs in every prompt; thinning induces unequal class counts.
    for prompt in range(prompts):
        for index, label in enumerate(LABELS):
            if rng.random() < 0.58 + 0.4 * index / 17:
                rows.append((len(rows), prompt, passage(rng, index, prompt), label))
    train = pd.DataFrame(rows, columns=['id', 'prompt_id', 'text', 'model'])
    test_rows = []
    for prompt in range(prompts, prompts + max(6, prompts // 4)):
        for index in range(18):
            test_rows.append((f'test-{len(test_rows)}', passage(rng, index, prompt)))
    test = pd.DataFrame(test_rows, columns=['id', 'text'])
    train.to_csv(out / 'train.csv', index=False)
    test.to_csv(out / 'test.csv', index=False)
    pd.DataFrame({'id': test.id, 'model': LABELS[0]}).to_parquet(out / 'sample_submission.parquet', index=False)
    return train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('data/synthetic'))
    parser.add_argument('--prompts', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train, test = generate(args.out, args.prompts, args.seed)
    print(f'Synthetic only: {len(train)} train / {len(test)} test; {train.prompt_id.nunique()} training prompts')
