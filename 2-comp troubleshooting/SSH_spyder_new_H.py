#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:24:36 2022

File for experimenting with Hamiltonians in SSH model in model defined as in 
Nguyen et al. https://doi.org/10.1038/s41534-020-0253-9

@author: k4cp3rskiii
"""
# =============================================================================
# File written for troubleshooting od 2-comp Hamiltonian code. 
# Discontinued at the moment. In order to work, it must be in 
# the same folder as 'Hubbard_aux.py' file, unless it gets developed in the 
# PIP-installable module by that time.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from numba.typed import List, Dict
import scipy as sc
import seaborn as sns
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
M = 40
# J - Hopping scaling factor
J = 1

# Number of basis vector components
component_count = 1
# Periodic bounary conditions
pbc = 0

# Statistic (bose/fermi)
stat_vec = np.array(['f'])

# V - On-site potential scaling factor
# V_vec_aa = np.array([0, 1])
# V_vec_bb = np.array([0, 1])

# V_vec_aa = np.array([0])
# V_vec_bb = np.array([0])

# Staggering tab - SSH parameter
v_vec = np.linspace(0, 2, 801)

#Disorder Measure - W_1, W_2
W_disorder_tab = np.array([0, 0])


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

vecs_dict = {}

Evals_tot_matrix = np.zeros((v_vec.shape[0], D))


for v_i in range(0, v_vec.shape[0]):
    
    
    delta = np.array([v_vec[v_i], 1])
    H_hop_1 = [List(), List(), List()]
    H_hop_1 = get_kinetic_H_vw(
        A,
        M,
        J,
        t_dict,
        vw=delta,
        disord_W=W_disorder_tab,
        pbc=pbc,
        statistic=stat[0],
        component_count=component_count,
        component_no=0,
    )
    H_hop_1 = sc.sparse.coo_matrix(
        (H_hop_1[0], (H_hop_1[1], H_hop_1[2])), shape=(D, D)
    )
    
    H_1 = J*H_hop_1
    
    
    H_tot = H_1
    

    evals, evecs = np.linalg.eigh(H_tot.toarray())
    idx = np.argsort(evals, -1)
    evecs = evecs[:, idx]
    evals = evals[idx]
    vecs_dict[v_i] = evecs.copy()
    evecs_dt = []
    evals_dt = []
    for evals_i in range(0, evals.shape[0]):
        psi = evecs[:, evals_i].copy()
        e_i = evals[evals_i]
        exp_val_H_tot = np.real(np.dot(np.conjugate(psi).T, np.dot(H_tot.toarray(), psi)))
        Evals_tot_matrix[v_i, evals_i] = exp_val_H_tot
        print("stat = " + str(stat) +
            ", v = {:.1f}, i = {:2d}, eval_i = {:.3f}".format(
                v_vec[v_i], evals_i, exp_val_H_tot
            )
        )
        
H_dense = H_tot.toarray()
        

figs, ax = plt.subplots(1, 1, figsize=(18, 10))
figs.tight_layout(pad=16)
figs.suptitle("stat="+str(stat)+" | N="+str(N)+" | J = {} |M={} | pbc = {}".format(J, M, pbc)+
              "\n"+r"$\hat{H}_{tot} = \sum_{i = 0}^{M/2-1} v (c^\dagger_{A_j} a_{B_j} + h.c) + w (c^\dagger_{A_{j+1}} a_{B_j} + h.c)$" + r"$, \quad \hat{H}_{total} | \lambda \rangle = E_{\lambda} | \lambda \rangle$", fontsize=20)
for i in range(0, Evals_tot_matrix.shape[1]):
    ax.set_title(r"$\langle \lambda | H_{total} | \lambda \rangle $", fontsize=18)
    ax.plot(v_vec, Evals_tot_matrix[:, i])
    ax.set_xlabel(r"$v$", fontsize=16)
    ax.set_ylabel(r"$E_{\lambda}$", fontsize=22)
    
figs.savefig("Eigenenergies_Plots/EE_plot_stat="+str(stat)+"_|_N="+str(N)+"_|_J={}_|_M={}_|_pbc={}".format(J, M, pbc)+".pdf")

#%%

# figs, axs = plt.subplots(1, 3, figsize=(15, 5))
# figs.tight_layout()
# show_wave_function(evecs[:, 2], A, plot_prob=False, plot_wf=True, bar=True, ax_apriori=axs[0], k=2)
# show_wave_function(evecs[:, 19], A, plot_prob=False, plot_wf=True, bar=True, ax_apriori=axs[1], k=19)
# show_wave_function(evecs[:, 20], A, plot_prob=False, plot_wf=True, bar=True, ax_apriori=axs[2], k=20)
# figs.show()
# figs.savefig("Edge_states_showcase.pdf")