import sys, os, time, pickle
sys.path.insert(0, '.')
from experiments.pipeline.run_fold import run_fold

datasets = ['diabetes', 'balance-scale', 'breast_cancer', 'car']
for ds_name in datasets:
    path = os.path.join('experiments', 'data', ds_name + '.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    X, y, splits = data['X'], data['y'], data['cv_splits']
    tr_idx, te_idx = splits[0]
    t0 = time.time()
    r = run_fold(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[te_idx], y.iloc[te_idx], mode='metacog', random_state=42)
    elapsed = time.time() - t0
    print(ds_name + ': ' + str(round(elapsed,2)) + 's  nodes=' + str(r['nodes']) + '  mcc=' + str(round(r['mcc'],4)))
