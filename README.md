# Name That LLM — Kaggle entry

Stylometric authorship-attribution pipeline for the Kaggle competition
[Name That LLM](https://www.kaggle.com/competitions/name-that-llm): predict which of
18 frontier LLMs wrote a 40–250 word passage.

> **Status / blocker:** the competition data (`train.csv`, `test.csv`) can only be
> downloaded after logging in to Kaggle and accepting the rules. No account was used
> here, so the pipeline was built and validated **on synthetic data only**. You need a
> Kaggle login to fetch the real data, train on it, and submit.

## Competition facts (from the competition pages)

| Item | Value |
|---|---|
| Task | 18-way classification of the authoring model |
| Train | `train.csv`: `id, prompt_id, text, model` — 2,699 passages, 150 prompts |
| Test | `test.csv`: `id, text` — ~4,000 hidden passages, **prompts unseen in train** |
| Metric | Balanced accuracy (mean per-class recall); class counts are unequal |
| Format | Code competition: notebook, internet off, ≤9 h, writes `submission.parquet` (`id, model`) |
| Deadlines | Entry/team merger Jan 6, 2027 23:59 UTC; final submission Jan 13, 2027 23:59 UTC |
| Limits | 5 submissions/day, 2 final selections, team size ≤5 |

## Approach

1. **Features** (`nthllm/features.py`, `nthllm/modeling.py`)
   - Char `char_wb` 1–5-gram TF-IDF (35k) and word 1–2-gram TF-IDF (18k), sublinear tf.
   - ~70 handcrafted style features:
     - lexical: type-token ratio, word/sentence length mean and std, long-word rate, function-word rates;
     - syntactic proxies: sentence count, sentence starters, passive and adverb proxies, contractions;
     - punctuation/formatting: em/en dash, semicolon, colon, curly vs straight quotes,
       parentheses, ellipsis, markdown bold/headings/bullets/numbered lists, newlines,
       digits, uppercase;
     - perplexity-style: character entropy, zlib compression ratio, word-frequency rank, rare-word rate.
   - Optional true LM perplexity from a local causal LM (e.g. GPT-2 attached as a Kaggle
     dataset, `--lm-path` / `NTHLLM_LM_PATH`). If it is missing, the feature is all zeros.
2. **Models:** class-balanced logistic regression, sigmoid-calibrated LinearSVC and log-loss
   SGD, combined by soft voting (averaged probabilities).
3. **Validation:** 5-fold `GroupKFold` by `prompt_id`, because test prompts are unseen.
   Inner SVC calibration is grouped by prompt too. Reports balanced accuracy and per-class recall.
4. **Submission:** checks columns, unique ids, nulls and the 18 valid labels, then writes `submission.parquet`.

## Layout

```
nthllm/                    features, models/CV, submission validation, label list
train.py / predict.py      CLI training (+CV report) and prediction
scripts/make_synthetic.py  synthetic data in the real schema (18 labels, imbalanced, 150 prompts)
scripts/build_notebook.py  regenerates the standalone Kaggle script
notebooks/kaggle_submission.py  self-contained offline Kaggle notebook script
tests/                     pytest suite
```

## Run locally

```bash
python -m pip install -r requirements.txt
python scripts/make_synthetic.py --out data/synthetic        # or put real Kaggle CSVs in data/real

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python train.py   --data-dir data/synthetic --out artifacts/model.joblib --folds 5
python predict.py --data-dir data/synthetic --model artifacts/model.joblib \
                  --out artifacts/submission.parquet --csv artifacts/submission.csv
NTHLLM_DATA_DIR=data/synthetic NTHLLM_OUT=artifacts/notebook_submission.parquet \
  python notebooks/kaggle_submission.py
python -m pytest -q
```

The CV report is written to `artifacts/model.cv.json`.

## Results

Synthetic data: 2,117 train passages, 150 prompts, 5-fold grouped CV, about 66 s on CPU.

| Model | Balanced accuracy |
|---|---:|
| Logistic regression | 0.698 ± 0.024 |
| Calibrated LinearSVC | 0.705 ± 0.018 |
| SGD (log loss) | 0.720 ± 0.018 |
| Soft-vote ensemble | 0.715 ± 0.022 |

These numbers only show that the pipeline works end to end. They say **nothing** about
leaderboard performance, because the synthetic styles are made up. Run `train.py` on the real
`train.csv` to get a real estimate. Expect real scores to be much lower than on synthetic data:
the 18 models include close siblings, e.g. several gpt-5.6 variants and gemini-3.x-flash versions.

## What you need to do to enter

1. Sign in to Kaggle, open the competition, click **Join Competition** and accept the rules
   before **Jan 6, 2027 23:59 UTC**.
2. Optional, for local tuning: download the data with
   `kaggle competitions download -c name-that-llm -p data/real`, unzip it, then run
   `python train.py --data-dir data/real --out artifacts/model.joblib`.
   Do not commit or redistribute the data; the rules forbid that.
3. Create a Kaggle notebook with the competition data attached and **internet off**. Paste
   `notebooks/kaggle_submission.py` into it and run it. It reads
   `/kaggle/input/competitions/name-that-llm/` and writes `submission.parquet`.
   Run `python scripts/build_notebook.py` first if you changed `nthllm/`.
4. Save/commit the notebook and click **Submit**. Pick your 2 final submissions before
   **Jan 13, 2027 23:59 UTC**.
5. Ideas for improving the score: tune C/alpha and n-gram ranges on the real CV; attach a
   small LM such as GPT-2 or Qwen-0.5B for perplexity features; fine-tune a DeBERTa-v3
   classifier on a GPU notebook and blend its probabilities with this ensemble.

## License

Code: MIT-style use permitted. Competition data is CC BY-SA 4.0 and is not included here.
