import math
import re
import warnings
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


FUNCTION_WORDS = 'the a an and or but if while although because to of in on for with by from is are was were be it this that we you'.split()
# A fixed reference vocabulary avoids fitting frequency statistics across CV folds.
REFERENCE = ('the of and to a in is it you that he was for on are as with his they at '
             'be this have from or one had by word but not what all were we when your '
             'can said there use an each which she do how their if will up other about '
             'out many then them these so some her would make like time has look two '
             'more write go see number no way could people my than first water been '
             'call who oil its now find long down day did get come made may part').split()
RANK = {word: i + 1 for i, word in enumerate(REFERENCE)}


class StyleFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [self._row(str(text)) for text in X]
        return np.asarray(rows, dtype=np.float64).reshape(-1, len(self.get_feature_names_out()))

    def get_feature_names_out(self, input_features=None):
        return np.array(self._row('', names=True))

    @staticmethod
    def _row(text, names=False):
        words = re.findall(r"\b[\w]+(?:['’][\w]+)?\b", text.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        n, c = max(len(words), 1), max(len(text), 1)
        lengths = [len(w) for w in words] or [0]
        sl = [len(s.split()) for s in sentences] or [0]
        counts = Counter(text)
        f = dict(word_count=len(words), ttr=len(set(words))/n,
                 word_length_mean=np.mean(lengths), word_length_std=np.std(lengths),
                 sentence_length_mean=np.mean(sl), sentence_length_std=np.std(sl),
                 sentence_count=len(sentences), long_word_rate=sum(x > 7 for x in lengths)/n,
                 entropy=-sum((v/c)*math.log2(v/c) for v in counts.values()),
                 compression=len(zlib.compress(text.encode()))/max(len(text.encode()), 1),
                 mean_rank=np.mean([RANK.get(w, 1000) for w in words] or [0]),
                 rare_rate=sum(w not in RANK for w in words)/n,
                 uppercase=sum(ch.isupper() for ch in text)/c,
                 digit_rate=sum(ch.isdigit() for ch in text)/c)
        for word in FUNCTION_WORDS:
            f['fw_' + word] = words.count(word)/n
        for key, pattern in {
            'em_dash':'—', 'en_dash':'–', 'semicolon':';', 'colon':':',
            'curly_quotes':'[“”‘’]', 'straight_quotes':'[\"\']',
            'parentheses':r'[()]', 'ellipsis':r'\.\.\.|…', 'bold':r'\*\*',
            'heading':r'(?m)^\s*#+ ', 'bullet':r'(?m)^\s*- ',
            'numbered':r'(?m)^\s*\d+[.)] ', 'newline':r'\n',
            'comma':',', 'question':r'\?', 'exclamation':'!',
            'contraction':r"\b\w+['’](?:t|s|re|ve|ll|d)\b",
            'passive_proxy':r'\b(?:is|was|were|been|be)\s+\w+ed\b',
            'adverb_proxy':r'\b\w+ly\b',
        }.items():
            f[key] = len(re.findall(pattern, text))/c
        for start in ('however', 'the', 'in', 'we', 'this', 'first', 'although'):
            f['starts_' + start] = sum(bool(re.match(start + r'\b', s.lower())) for s in sentences)/max(len(sentences), 1)
        return list(f) if names else list(f.values())


class LocalPerplexity(BaseEstimator, TransformerMixin):
    """Optional offline causal-LM feature; missing dependencies yield a zero column."""

    def __init__(self, model_path=None, max_tokens=256):
        self.model_path = model_path
        self.max_tokens = max_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = list(X)
        result = np.zeros((len(texts), 1))
        if not self.model_path:
            return result
        try:
            if not Path(self.model_path).is_dir():
                raise FileNotFoundError(self.model_path)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True).eval()
            for i, text in enumerate(texts):
                tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_tokens)
                if tokens['input_ids'].shape[1] > 1:
                    with torch.no_grad():
                        loss = model(**tokens, labels=tokens['input_ids']).loss.item()
                    result[i, 0] = math.exp(min(loss, 20)) if math.isfinite(loss) else 0
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            warnings.warn(f'Local perplexity unavailable; using zeros: {exc}', RuntimeWarning)
        return result
