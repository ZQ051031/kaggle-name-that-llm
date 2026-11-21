import pandas as pd

from nthllm import LABELS


def validate_submission(submission, test):
    if list(submission.columns) != ['id', 'model']:
        raise ValueError('Submission columns must be id, model in that order')
    if submission.isna().any().any() or test.id.isna().any():
        raise ValueError('Null submission values or test IDs')
    if submission.id.duplicated().any() or test.id.duplicated().any():
        raise ValueError('Duplicate IDs')
    if len(submission) != len(test) or set(submission.id) != set(test.id):
        raise ValueError('Submission IDs must match test exactly')
    if not submission.model.isin(LABELS).all():
        raise ValueError('Invalid model labels')
    return submission


def write_submission(test, predictions, out, csv=None):
    submission = validate_submission(pd.DataFrame({'id': test.id, 'model': predictions}), test)
    submission.to_parquet(out, index=False)
    if csv:
        submission.to_csv(csv, index=False)
    return submission
