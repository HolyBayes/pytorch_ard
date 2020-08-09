import sys
sys.path.append('../')
from models import DenseModelARD
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from torch_ard import get_ard_reg, get_dropped_params_ratio, ELBOLoss
from tqdm import trange, tqdm

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target
X, y = df.drop('PRICE', 1), df['PRICE']

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
train_X, test_X, train_y, test_y = \
    [torch.from_numpy(np.array(x)).float().to(device)
     for x in [train_X, test_X, train_y, test_y]]

model = DenseModelARD(input_shape=train_X.shape[1], output_shape=1,
                      activation=nn.functional.relu).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
criterion = ELBOLoss(model, F.mse_loss).to(device)

n_epoches = 100000
debug_frequency = 100
def get_kl_weight(epoch): return min(1, 2 * epoch / n_epoches)


pbar = trange(n_epoches, leave=True, position=0)
for epoch in pbar:
    kl_weight = get_kl_weight(epoch)
    opt.zero_grad()
    preds = model(train_X).squeeze()
    loss = criterion(preds, train_y, 1, kl_weight)
    loss.backward()
    opt.step()
    loss_train = float(
        criterion(preds, train_y, 1, 0).detach().cpu().numpy())
    preds = model(test_X).squeeze()
    loss_test = float(
        criterion(preds, test_y, 1, 0).detach().cpu().numpy())
    pbar.set_description('MSE (train): %.3f\tMSE (test): %.3f\tReg: %.3f\tDropout rate: %f%%' % (
        loss_train, loss_test, get_ard_reg(model).item(), 100 * get_dropped_params_ratio(model)))
    pbar.update()
