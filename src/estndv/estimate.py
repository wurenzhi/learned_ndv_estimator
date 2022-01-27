import os

import numpy as np


def to_sparse(F):
    '''
    convert dense F to sparse representation
    :param F:
    :return:
    '''
    F = np.array(F).astype(int)
    F_ind = np.arange(1, len(F) + 1)
    keep = np.where(F > 0)
    F_ind = F_ind[keep]
    F_val = F[keep]
    return np.array([F_ind, F_val]).T


def feature_eng_sparse(fs, Ns):
    '''
    feature engineering according to Section 4.2.1
    :param fs:
    :param Ns:
    :return:
    '''
    Ns = np.array(Ns)
    m = 100
    ndv_s = np.array([np.sum(f[:, 1]) for f in fs])
    n_s = np.array([np.sum(f[:, 0] * f[:, 1]) for f in fs])
    fe1 = [Ns.reshape(-1, 1), n_s.reshape(-1, 1), Ns.reshape(-1, 1) / (n_s.reshape(-1, 1) + 1e-6),
           ndv_s.reshape(-1, 1)]
    f_s_trunc = []
    n_s_truncated = []
    ndv_truncated = []
    for i_fs in fs:
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
    return X, ndv_truncated


class ndvEstimator:
    def __init__(self, para_path=None):
        '''
        :param para_path: if None, the default pretrained model parameters are used
        '''
        if para_path is None:
            pt = os.path.dirname(os.path.realpath(__file__))
            para_path = os.path.join(pt,"model_paras.npy")
        self.paras = np.load(para_path, allow_pickle=True)

    def sample2profile(self, S):
        _, counts = np.unique(S, return_counts=True)
        cnt, val = np.unique(counts, return_counts=True)
        fs = np.zeros(np.max(cnt)).astype(int)
        fs[cnt - 1] = val
        return fs

    def sample_predict(self, S, N):
        """
        estimate number of distinct values in population from a sample S
        :param S: a sample, list
        :param N: population size, int
        :return: estimated number of distinct values, int
        """
        f = self.sample2profile(S)
        f_sparse = to_sparse(f)
        ndv = self.profile_predict_batch([f_sparse], [N], is_sparse=True)
        return ndv

    def profile_predict(self, f, N):
        """
        estimate number of distinct values in population from a sample profile f
        :param f: a sample profile, list
        :param N: population size, int
        :return: estimated number of distinct values, int
        """
        f_sparse = to_sparse(f)
        ndv = self.profile_predict_batch([f_sparse], [N], is_sparse=True)
        return ndv

    def sample_predict_batch(self, S_list, N_list):
        """
        estimate number of distinct values in batch using samples
        :param S_list: list of samples
        :param N_list: list of population size
        :return: list of estimated ndv
        """
        fs = [self.sample2profile(s) for s in S_list]
        fs_sparse = [to_sparse(f) for f in fs]
        ndvs = self.profile_predict_batch(fs_sparse, N_list, is_sparse=True)
        return ndvs

    def profile_predict_batch(self, f_list, N_list, is_sparse=False):
        """
        estimate number of distinct value of population from sample profile
        :param fs: list of sample profile
        :param Ns: list of population size
        :param is_sparse: is the profile in sparse representation?
        :return: list of estimated ndv
        """
        if not is_sparse:
            f_list = [to_sparse(item) for item in f_list]
        x, truncated = feature_eng_sparse(f_list, N_list)
        n_layers = 7
        for i in range(n_layers):
            w = self.paras[2 * i]
            b = self.paras[2 * i + 1]
            x = np.dot(x, w.T) + b.reshape(1, -1)
            if i <= 4:
                y1 = ((x > 0) * x)
                y2 = ((x <= 0) * x * 0.01)
                x = y1 + y2
        y_p = np.exp(x) - 0.1 + truncated
        y_p = np.squeeze(y_p)
        return y_p


if __name__ == '__main__':
    estimator = ndvEstimator(para_path="model_paras.npy")
    D1 = estimator.profile_predict(f=[10 ** 3, 0], N=10 ** 4)
    D2 = estimator.profile_predict(f=[10 ** 13, 0], N=10 ** 14)
    print(D1, D2)
    print(estimator.sample_predict_batch(S_list=[[1, 1, 1, 3, 5, 5, 12], [1, 1, 1, 1, 5, 5, 12]], N_list=[100000, 500]))
    print(estimator.profile_predict_batch(f_list=[[2, 1, 1]], N_list=[100000]))
    print(estimator.profile_predict_batch(f_list=[[2, 1, 1], [1, 1, 0, 1]], N_list=[100000, 500]))
    for i in range(9, 20):
        print(i, estimator.profile_predict(f=[10 ** i, 0], N=int(10 ** (i + 1))))
