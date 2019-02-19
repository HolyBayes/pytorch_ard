import sys
sys.path.append('../')
from models import DenseModelARD
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import pandas as pd
from torch import nn
import torch
import numpy as np
from torch_ard import get_ard_reg, get_dropped_params_ratio

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target
X, y = df.drop('PRICE', 1), df['PRICE']

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
train_X, test_X, train_y, test_y = \
    [torch.from_numpy(np.array(x)).float().to(device) for x in [train_X, test_X, train_y, test_y]]

model = DenseModelARD(input_shape=train_X.shape[1], output_shape=1,
        activation=nn.functional.relu).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

n_epoches = 100000
debug_frequency = 100
reg_factor = 1e-3

for epoch in range(n_epoches):
    opt.zero_grad()
    preds = model(train_X).squeeze()
    reg = get_ard_reg(model)
    loss = criterion(preds, train_y) + reg*reg_factor
    loss.backward()
    opt.step()
    loss_train = float(criterion(preds, train_y).detach().cpu().numpy())
    preds = model.predict(test_X, deterministic=True).squeeze()
    loss_test = float(criterion(preds, test_y).detach().cpu().numpy())
    if epoch % debug_frequency == 0:
        print('%d epoch' % epoch)
        print('MSE (train): %.3f' % loss_train)
        print('MSE (test): %.3f' % loss_test)
        print('Reg: %.3f' % reg.item())
        print('Dropout rate: %f%%' % (100*get_dropped_params_ratio(model)))
