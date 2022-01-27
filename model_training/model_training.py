from os.path import join
import numpy as np
from skorch.callbacks import LoadInitState, TrainEndCheckpoint
from skorch.dataset import Dataset
import skorch
import torch
from skorch.helper import predefined_split
from sklearn.model_selection import train_test_split
from skorch import NeuralNetRegressor
from torch import optim
from network import Regressor, Loss_gamma_0_6
import multiprocessing


def feature_eng_sparse(f_s, n, m, ndv=None):
    '''
    feature engineering according to Section 4.2.1
    :param f_s:
    :param n:
    :param m:
    :param ndv:
    :return:
    '''
    ndv_s = np.array([np.sum(f[:, 1]) for f in f_s])
    n_s = np.array([np.sum(f[:, 0] * f[:, 1]) for f in f_s])
    fe1 = [n.reshape(-1, 1), n_s.reshape(-1, 1), n.reshape(-1, 1) / (n_s.reshape(-1, 1) + 1e-6),
           ndv_s.reshape(-1, 1)]
    f_s_trunc = []
    n_s_truncated = []
    ndv_truncated = []
    for i_fs in f_s:
        tmp1 = i_fs[i_fs[:, 0] < m, :]
        tmp2 = i_fs[i_fs[:, 0] >= m, :]
        i_fs_trunc = np.zeros(m)
        i_fs_trunc[tmp1[:, 0] - 1] = tmp1[:, 1]
        f_s_trunc.append(i_fs_trunc)
        ndv_truncated.append(np.sum(tmp2[:, 1]))
        n_s_truncated.append(np.sum(tmp2[:, 1] * tmp2[:, 0]))
    ndv_truncated = np.array(ndv_truncated).reshape(-1, 1)
    n_s_truncated = np.array(n_s_truncated).reshape(-1, 1)
    f_s_trunc = np.array(f_s_trunc)
    fe2 = [n_s_truncated, ndv_truncated, f_s_trunc]
    X = np.concatenate(fe1 + fe2, axis=1)
    X = np.log(np.abs(X) + 1e-3)
    print(X.shape)
    if ndv is not None:
        y = ndv - ndv_truncated
        y = np.log(y + 0.1)
        return X, y
    else:
        return X


class ndvDatasetSparse(Dataset):
    def __init__(self, f_path, f_s_path, X):
        self.f = np.load(f_path)
        self.f_s = np.load(f_s_path, allow_pickle=True)
        self.ndv = self.f[:, 1]
        self.n = self.f[:, 0]
        assert np.sum(self.ndv > self.n) == 0
        self.m = 100

        self.ndv = self.ndv.reshape(-1, 1)
        self.X, self.y = feature_eng_sparse(self.f_s, self.n, self.m, self.ndv)
        n_sample = np.array([np.sum(fs[:, 0] * fs[:, 1]) for fs in self.f_s])
        self.r = n_sample / (self.n + 1e-6)
        self.r = self.r.reshape(-1, 1)
        self.w = np.ones_like(self.r)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, :], self.y[idx]


def load_training_dataset(data_path="training_data"):
    data = ndvDatasetSparse(join(data_path, "rfs_F_infos.npy"),
                            join(data_path, "rfs_f_s.npy"))
    print("data loaded")
    X_train, X_test, y_train, y_test, w_train, w_test, r_train, r_test = train_test_split(data.X.astype(np.float32),
                                                                                          data.y.astype(np.float32),
                                                                                          data.w.astype(np.float32),
                                                                                          data.r.astype(np.float32),
                                                                                          test_size=0.1,
                                                                                          random_state=42)

    return X_train, X_test, y_train, y_test, w_train, w_test, r_train, r_test


def model_training(cp_name, description, loss, Network=Regressor, resume_cp=None, lr=0.0003, wd=0.1,device='cuda'):
    print(cp_name)
    print(description)
    X_train, X_val, y_train, y_val, w_train, w_val, r_train, r_val = \
        load_training_dataset()
    cp = skorch.callbacks.Checkpoint(dirname=cp_name)
    X_val_obj = {'data': X_val, 'sample_weight': w_val, 'sample_rate': r_val}
    valid_ds = Dataset(X_val_obj, y_val)
    X_train_obj = {'data': X_train, 'sample_weight': w_train, 'sample_rate': r_train}
    train_ds = Dataset(X_train_obj, y_train)
    train_end_cp = TrainEndCheckpoint(dirname=cp_name + "_train_end")
    n_workers = multiprocessing.cpu_count()
    if resume_cp is not None:
        rcp = skorch.callbacks.Checkpoint(dirname=resume_cp)
        load_state = LoadInitState(rcp)
        net = NeuralNetRegressor(
            Network(n_in=X_train.shape[1]),
            criterion=loss,
            max_epochs=2000,
            optimizer=optim.Adam,
            optimizer__amsgrad=True,
            optimizer__weight_decay=wd,
            lr=lr,  # 0.000003,
            iterator_train__shuffle=True,
            iterator_train__num_workers=n_workers,
            iterator_train__pin_memory=True,
            device=device,
            batch_size=50000,
            iterator_train__batch_size=50000,
            train_split=predefined_split(valid_ds),
            callbacks=[
                cp,
                load_state,
                train_end_cp,
            ]
        )
    else:
        net = NeuralNetRegressor(
            Network(n_in=X_train.shape[1]),
            criterion=loss,
            max_epochs=2000,
            optimizer=optim.Adam,
            optimizer__amsgrad=True,
            optimizer__weight_decay=wd,
            lr=lr,  # 0.000003,
            iterator_train__shuffle=True,
            iterator_train__num_workers=n_workers,
            iterator_train__pin_memory=True,
            device=device,
            batch_size=50000,
            iterator_train__batch_size=50000,
            train_split=predefined_split(valid_ds),
            callbacks=[
                cp,
                train_end_cp,
            ]
        )
    net.fit(train_ds, y=None)
    if resume_cp is None:
        dir = "./"
        with open(join(dir, join(cp_name, "desc.txt")), "w") as text_file:
            print(description, file=text_file)


model_training("cp_gamma_0_6", description="gamma=0.6", loss=Loss_gamma_0_6, device='cuda')
