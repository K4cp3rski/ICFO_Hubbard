#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:24:36 2022

File for experimenting with Hamiltonians in SSH model in model defined as in 
Nguyen et al. https://doi.org/10.1038/s41534-020-0253-9

@author: k4cp3rskiii
"""


import numpy as np
import matplotlib.pyplot as plt
from numba.typed import List, Dict
import scipy as sc
import pathlib

from Hubbard_aux import (
    calc_Dim,
    calc_dim_tab,
    show_wave_function,
    get_basis,
    get_tensor_basis,
    a_operator,
    tag_func,
    find_orig_idx,
    get_kinetic_H,
    get_kinetic_H_vw,
    get_interactions_H,
    get_intercomp_interactions_H,
    plot_coo_matrix,
    save_sparse_coo,
    load_sparse_coo,
    fock_dot,
    get_density_matrix_comp,
    get_density_matrix,
    get_prob_mat
)

# %%
# N - Number of particles in a system
N = np.array([1])  # If number - the same number for both systems
# N = np.array([1, 1])  # If list - [n_1, n_2] - subsystem sizes
#
# M - Number of sites to fill
M = 10
# J - Hopping scaling factor
J = 1

# Number of basis vector components
component_count = 1
# Periodic bounary conditions
pbc = 1

# Statistic (bose/fermi)
stat_vec = np.array(['f'])

# V - On-site potential scaling factor
# V_vec_aa = np.array([0, 1])
# V_vec_bb = np.array([0, 1])

# V_vec_aa = np.array([0])
# V_vec_bb = np.array([0])

# Staggering tab - SSH parameter
# delta_vec = np.linspace(-0.9, 0.9, 40)
vw_tab = np.array([1, 1])

#Disorder Measure - W_1, W_2
W_disorder_tab = np.array([1, 1])

p_plot = (
    pathlib.Path.cwd()
    .joinpath("Eigenenergies_Plots")
)
if not p_plot.exists():
    p_plot.mkdir(parents=True, exist_ok=True)

stat = stat_vec


# V - On-site potential scaling factor
# V = np.array([0])
# V = np.array([1, 2])
# Statistic (bose/fermi)
# stat = np.array(['b'])  # If single - the same statistic in both
# stat = np.array(["f", "f"])  # If list - [stat_1, stat_2] - subsystem statistics
# Staggering - SSH parameter
# delta_t = 0.0
# Intercomponent interaction strength
# U = 0


# D - Dimension of the final Hamiltonian matrix
D = calc_dim_tab(M, N, stat_tab=stat)
# D = 10
A = get_tensor_basis(M, N, statistic=stat, component_count=component_count, verb=0)
# print(A.shape)

# Getting the basis vectors hashed
tab_T = np.array([tag_func(v) for v in A])
# Preserving the original order of basis vectors
ind = np.argsort(tab_T)
# Sorting the new array for more efficient searching (by bisection)
t_sorted = tab_T.copy()
t_sorted.sort()
t_dict = Dict()

for key, val in zip(tab_T, np.arange(0, A.shape[0])):
    t_dict[key] = val



H_hop_1 = [List(), List(), List()]
H_hop_1 = get_kinetic_H_vw(
    A,
    M,
    J,
    t_dict,
    vw=vw_tab,
    disord_W=W_disorder_tab,
    pbc=pbc,
    statistic=stat[0],
    component_count=component_count,
    component_no=0,
)
H_hop_1 = sc.sparse.coo_matrix(
    (H_hop_1[0], (H_hop_1[1], H_hop_1[2])), shape=(D, D)
)

H_hop_1 = sc.sparse.triu(H_hop_1)

H_hop_1 = H_hop_1 + H_hop_1.T - sc.sparse.diags(H_hop_1.diagonal(), format='coo')

H_1 = J*H_hop_1


H_tot = H_1

H_dense = H_tot.toarray()

