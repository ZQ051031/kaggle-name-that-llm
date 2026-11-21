import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from nthllm import LABELS
from nthllm.features import LocalPerplexity, StyleFeatures


def make_features(lm_path=None):
    return FeatureUnion([
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5),
                                sublinear_tf=True, max_features=35000)),
        ('word', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                max_features=18000, token_pattern=r'(?u)\b\w+\b')),
        ('style', Pipeline([('extract', StyleFeatures()), ('scale', MaxAbsScaler())])),
        ('lm', Pipeline([('extract', LocalPerplexity(lm_path)), ('scale', MaxAbsScaler())])),
    ])


def make_models():
    return {
        'logistic': LogisticRegression(C=4, class_weight='balanced', max_iter=600),
        # Calibration is fitted inside each outer training fold, never on held-out prompts.
        'calibrated_svc': CalibratedClassifierCV(
            LinearSVC(C=0.7, class_weight='balanced', dual='auto'), method='sigmoid'),
        'sgd': SGDClassifier(loss='log_loss', alpha=0.001, class_weight='balanced',
                             max_iter=1000, tol=1e-4, random_state=42),
    }


def probabilities(models, X):
    result = {}
    for name, model in models.items():
        aligned = np.zeros((X.shape[0], len(LABELS)))
        raw = model.predict_proba(X)
        for column, label in enumerate(model.classes_):
            aligned[:, LABELS.index(label)] = raw[:, column]
        result[name] = aligned
    result['ensemble'] = np.mean(list(result.values()), axis=0)
    return result


def train(data, folds=5, lm_path=None):
    required = {'id', 'prompt_id', 'text', 'model'}
    if not required.issubset(data.columns) or data[list(required)].isna().any().any():
        raise ValueError('Training data requires non-null id, prompt_id, text, model')
    if set(data.model) != set(LABELS):
        raise ValueError('Training data must contain exactly the 18 competition labels')
    if data.id.duplicated().any():
        raise ValueError('Duplicate training IDs')
    start = time.perf_counter()
    scores = {name: [] for name in [*make_models(), 'ensemble']}
    oof = {name: np.empty(len(data), dtype=object) for name in scores}
    split_report = []
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(data, data.model, data.prompt_id), 1):
        if set(data.iloc[tr].model) != set(LABELS):
            raise ValueError('Every training fold must contain all labels; use more prompts')
        features = make_features(lm_path)
        X = features.fit_transform(data.iloc[tr].text)
        V = features.transform(data.iloc[va].text)
        models = make_models()
        # Inner calibration also respects prompt groups.
        inner = list(GroupKFold(3).split(X, data.iloc[tr].model, data.iloc[tr].prompt_id))
        models['calibrated_svc'].cv = inner
        for model in models.values():
            model.fit(X, data.iloc[tr].model)
        for name, probs in probabilities(models, V).items():
            pred = np.array(LABELS)[probs.argmax(axis=1)]
            oof[name][va] = pred
            score = balanced_accuracy_score(data.iloc[va].model, pred)
            scores[name].append(float(score))
            print(f'fold={fold} {name}: {score:.4f}', flush=True)
        split_report.append({'fold': fold, 'train_prompts': sorted(map(str, data.iloc[tr].prompt_id.unique())),
                             'valid_prompts': sorted(map(str, data.iloc[va].prompt_id.unique()))})
    report = {'models': {}, 'splits': split_report}
    for name, values in scores.items():
        report['models'][name] = {'fold_scores': values, 'mean': float(np.mean(values)),
                                  'std': float(np.std(values)), 'per_class_recall': dict(zip(
                                      LABELS, recall_score(data.model, oof[name], labels=LABELS,
                                                           average=None, zero_division=0).tolist()))}
    features = make_features(lm_path)
    X = features.fit_transform(data.text)
    models = make_models()
    models['calibrated_svc'].cv = list(GroupKFold(3).split(X, data.model, data.prompt_id))
    for model in models.values():
        model.fit(X, data.model)
    report['runtime_seconds'] = time.perf_counter() - start
    return {'features': features, 'models': models, 'labels': LABELS}, report


def predict(artifact, texts):
    X = artifact['features'].transform(texts)
    return np.array(LABELS)[probabilities(artifact['models'], X)['ensemble'].argmax(axis=1)]
