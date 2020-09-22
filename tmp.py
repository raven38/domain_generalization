import numpy as np
from pathlib import Path


def softmax(x, axis=None):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def log_softmax(x, axis=None):
    return np.log(softmax(x, axis))
    
def calc_scores(stats_file, preds_file):
    mu_hat = np.load(stats_file)['mu_hat']
    sigma_hat = np.load(stats_file)['sigma_hat']

    inv_s_hat = np.linalg.inv(sigma_hat)
    preds = np.load(preds_file)
    features = preds['feature']
    ys = preds['y']
    domains = preds['domain']
    outputs = preds['output']

    ed = np.sqrt(np.sum((features - mu_hat) * (features - mu_hat), axis=1))
    thresholds = np.percentile(ed, list(range(101)))
    ed_acc = 0
    for th in thresholds:
        unknown = ((ed > th).astype(float) * (outputs.max() + 1e-8)).reshape(-1, 1)
        corrects = np.concatenate([outputs, unknown], axis=1).argmax(axis=1) == ys
        # print(corrects.mean())
        ed_acc = max(ed_acc, corrects[domains==0].mean())

    md = np.diag((features - mu_hat) @ inv_s_hat @ (features - mu_hat).T)
    thresholds = np.percentile(md, list(range(101)))
    md_acc = 0
    for th in thresholds:
        unknown = ((md > th).astype(float) * (outputs.max() + 1e-8)).reshape(-1, 1)
        corrects = np.concatenate([outputs, unknown], axis=1).argmax(axis=1) == ys
        # print(corrects.mean())
        md_acc = max(md_acc, corrects[domains==0].mean())


    entropy = -np.sum(softmax(outputs, 1) * log_softmax(outputs, 1), 1)
    thresholds = np.percentile(entropy, list(range(101)))
    h_acc = 0
    for th in thresholds:
        unknown = ((entropy > th).astype(float) * (outputs.max() + 1e-8)).reshape(-1, 1)
        corrects = np.concatenate([outputs, unknown], axis=1).argmax(axis=1) == ys
        # print(corrects.mean())
        h_acc = max(h_acc, corrects[domains==0].mean())

    acc = (outputs.argmax(axis=1) == ys)[domains==0].mean()
    print(f'{stats_file.parent}/{stats_file.stem.split("_")[0]} closet set acc {acc}, mahalanobis acc {md_acc}, euclid acc {ed_acc}, entropy acc {h_acc}')

stats_files = list(sorted(Path('predicts').glob('*/*_mahalanobis_statistics.npz')))
preds_files = list(sorted(Path('predicts').glob('*/*_predicts.npz')))

stats_file = stats_files[4]
preds_file = preds_files[4]
for stats_file, preds_file in zip(stats_files, preds_files):
    calc_scores(stats_file, preds_file)
