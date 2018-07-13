from model import DenseModel
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import pandas as pd
from torch import nn
import torch
import numpy as np

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target
X, y = df.drop('PRICE', 1), df['PRICE']

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)

train_X, test_X, train_y, test_y = \
    [torch.from_numpy(np.array(x)).float() for x in [train_X, test_X, train_y, test_y]]

model = DenseModel(input_shape=train_X.shape[1], output_shape=1, activation=nn.functional.relu)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

n_epoches = 10000
debug_frequency = 100
reg_factor = 1.0

for epoch in range(n_epoches):
    opt.zero_grad()
    preds = model(train_X).squeeze()
    loss = criterion(preds, train_y) + reg_factor * model.eval_reg()
    loss.backward()
    opt.step()
    loss_train = float(criterion(preds, train_y).detach().cpu().numpy())
    preds = model.predict(test_X, deterministic=True).squeeze()
    loss_test = float(criterion(preds, test_y).detach().cpu().numpy())
    if epoch % debug_frequency == 0:
        print('%d epoch' % epoch)
        print('MSE (train): %.3f' % loss_train)
        print('MSE (test): %.3f' % loss_test)
        print('Compression: %.2f%%' % (100 * model.eval_compression()))
