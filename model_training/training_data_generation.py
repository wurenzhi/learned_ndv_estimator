import math
import multiprocessing
import random
from os.path import join
import os

import numpy as np
from tqdm import tqdm
from functools import partial


def sampling_from_profile_sparse(F_sparse, sampling_rate):
    '''
    generate a sampling profile from a population profile F_sparse using the method in Section 4.1.2
    :param F_sparse:
    :param sampling_rate:
    :return:
    '''
    fs = []
    for j in range(F_sparse.shape[0]):
        k = F_sparse[j, 0]
        F_k = F_sparse[j, 1]
        f_i = np.random.binomial(k, sampling_rate, size=F_k)
        fs.append(f_i)
    fs = np.concatenate(fs)
    fs = fs[fs > 0]
    unique, counts = np.unique(fs, return_counts=True)
    f_sparse = np.array([unique, counts]).T
    return f_sparse


def to_sparse(F):
    '''
    convert dense F to sparse representation
    :param F:
    :return:
    '''
    F = F.astype(int)
    F_ind = np.arange(1, len(F) + 1)
    keep = np.where(F > 0)
    F_ind = F_ind[keep]
    F_val = F[keep]
    return np.array([F_ind, F_val]).T


def rfs():
    '''
    generate one population profile using the method in Section 4.1.1
    :return:
    '''
    N = int(math.pow(10.0, random.random() * 5 + 4))  # One random population size N in [10^4,10^9]
    M = np.power(10, np.random.random() * 7).astype(int)  # length of the pop profile, set to 7 due to memory limit

    ##### random fixed sum algorithm
    #### https://stackoverflow.com/a/8064754
    t = np.random.randint(0, N + 1, M + 1)
    t[0] = 0
    t[-1] = N
    g = np.sort(np.diff(np.sort(t)))
    F = np.zeros(len(g))
    F[:-1] = np.flip(np.diff(g))
    F[-1] = g[0]
    F = F.astype(int)
    ####

    ### diversify Section 4.1.3
    N_p = math.pow(10.0, random.random() * 5 + 4)
    i_p = np.power(10, np.random.random() * 7).astype(int) #set to 7 due to memory limit
    D_p = int(N_p / i_p)
    if D_p == 0:
        F_sparse = to_sparse(F)
    elif i_p <= M:
        F[i_p - 1] += D_p
        F_sparse = to_sparse(F)
    else:
        F_sparse = to_sparse(F)
        F_h = np.array([i_p, D_p]).reshape(1, -1)
        F_sparse = np.concatenate([F_sparse, F_h])
    return F_sparse


def generate_training_data(seed, mtd, n_data):
    random.seed(seed)
    np.random.seed(seed)
    f_s_list = []
    p_sizes = []
    ndvs = []
    pbar = tqdm(total=n_data)
    while len(p_sizes) < n_data:
        rate = math.pow(10.0, random.random() * (-3) - 1)
        F_sparse = mtd()
        if F_sparse.shape[0] == 0:
            continue
        f_sparse = sampling_from_profile_sparse(F_sparse, rate)
        if f_sparse.shape[0] == 0:
            continue
        p_sizes.append(np.sum(F_sparse[:, 0] * F_sparse[:, 1]))
        ndvs.append(np.sum(F_sparse[:, 1]))
        f_s_list.append(f_sparse)
        pbar.update(1)
    F_info = np.array([p_sizes, ndvs]).T
    return F_info, f_s_list



if __name__ == "__main__":
    n_jobs = multiprocessing.cpu_count()
    seeds = np.arange(n_jobs).astype(int).tolist()
    mtd, name = rfs, "rfs"
    print(name)
    pool = multiprocessing.Pool(n_jobs)
    gen = partial(generate_training_data, mtd=mtd, n_data=10)
    result = pool.map(gen, seeds)
    pool.close()
    pool.terminate()
    pool.join()
    F_infos = []
    f_s_list = []
    for F_info, f_s in tqdm(result):
        F_infos.append(F_info)
        f_s_list.extend(f_s)
    F_infos = np.concatenate(F_infos)
    data_path = "training_data"
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    np.save(join(data_path, name + "_F_infos.npy"), F_infos)
    np.savez(join(data_path, name + "_f_s"), *f_s_list)
