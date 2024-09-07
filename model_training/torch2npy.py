from network import Regressor, Loss_gamma_0_6
import numpy as np
import skorch
from skorch import NeuralNetRegressor
from torch import optim


def load_model(load_cp, n_in=106,device='cuda'):
    cp = skorch.callbacks.Checkpoint(dirname=load_cp)
    net = NeuralNetRegressor(
        Regressor(n_in=n_in),
        criterion=Loss_gamma_0_6,
        max_epochs=2000,
        optimizer=optim.Adam,
        optimizer__amsgrad=True,
        optimizer__weight_decay=0.1,
        lr=0.0003,
        iterator_train__shuffle=True,
        iterator_train__num_workers=32,
        iterator_train__pin_memory=True,
        device=device,
        batch_size=50000,
        iterator_train__batch_size=50000,
    )
    net.initialize()
    net.load_params(checkpoint=cp)
    return net


def save_model_para(model_cp):
    '''
    convert trained model paras(saved at checkpoint model_cp) to numpy format
    :param model_cp:
    :return:
    '''
    model = load_model(model_cp, n_in=106,device='cpu')
    paras = []
    for para in model.get_params()['module'].parameters():
        paras.append(para.data.cpu().numpy())
    np.save("model_paras.npy", np.array(paras, dtype=object), allow_pickle=True)

save_model_para('cp_gamma_0_6')
