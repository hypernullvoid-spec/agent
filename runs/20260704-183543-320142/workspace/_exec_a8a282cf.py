import numpy as np, pandas as pd
df = pd.read_csv('./input/train.csv')
X = df[['f1','f2','f3']].values; y = df['target'].values
n = len(df); cut = int(n*0.8)
Xtr, Xva, ytr, yva = X[:cut], X[cut:], y[:cut], y[cut:]
Xb = np.c_[np.ones(cut), Xtr]
w = np.linalg.lstsq(Xb, ytr, rcond=None)[0]
pred = np.c_[np.ones(n-cut), Xva] @ w
rmse = float(np.sqrt(((pred - yva)**2).mean()))
print(f"Final Validation Metric: {rmse:.4f}")