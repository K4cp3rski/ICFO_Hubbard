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
    get_interactions_H,
    get_intercomp_interactions_H,
    plot_coo_matrix,
    save_sparse_coo,
    load_sparse_coo,
    fock_dot,
    get_density_matrix_comp,
    get_density_matrix,
)

# %%
# N - Number of particles in a system
# N = np.array([2])  # If number - the same number for both systems
N = np.array([1, 1])  # If list - [n_1, n_2] - subsystem sizes
#
# M - Number of sites to fill
M = 6
# J - Hopping scaling factor
J = 1

# Number of basis vector components
component_count = 2
# Periodic bounary conditions
pbc = 0

# Statistic (bose/fermi)
stat_a_vec = np.array(['f'])
stat_b_vec = np.array(['f'])

# V - On-site potential scaling factor
# V_vec_aa = np.array([0, 1])
# V_vec_bb = np.array([0, 1])

V_vec_aa = np.array([0])
V_vec_bb = np.array([0])

# Staggering tab - SSH parameter
delta_vec = np.linspace(-0.9, 0.9, 40)


# Intercomponent interaction strength values tab
U_vec = np.array([0])#([0, 0.001, 0.01, 0.1, 1])

p_plot = (
    pathlib.Path.cwd()
    .joinpath("Eigenenergies_Plots")
)
if not p_plot.exists():
    p_plot.mkdir(parents=True, exist_ok=True)

for stat_i in range(0, stat_a_vec.shape[0]):
    stat = np.array([stat_a_vec[stat_i], stat_b_vec[stat_i]])
    
    
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
    
    
    for U_i in range(0, U_vec.shape[0]):
        for V_i in range(0, V_vec_aa.shape[0]):
            V = np.array([V_vec_aa[V_i], V_vec_bb[V_i]])
            U = U_vec[U_i]
            
            
            H_aa = [List(), List(), List()]
            H_aa = get_interactions_H(A, M, 1, t_dict, statistic=stat[0], component_count=component_count, component_no=0, pbc=pbc)
            H_aa = sc.sparse.coo_matrix((H_aa[0], (H_aa[1], H_aa[2])), shape=(D, D))
            
            H_bb = get_interactions_H(A, M, 1, t_dict, statistic=stat[1], component_count=component_count, component_no=1, pbc=pbc)
            H_bb = sc.sparse.coo_matrix((H_bb[0], (H_bb[1], H_bb[2])), shape=(D, D))
            
            Evals1_matrix = np.zeros((delta_vec.shape[0], D))
            Evals2_matrix = np.zeros((delta_vec.shape[0], D))
            Evals_tot_matrix = np.zeros((delta_vec.shape[0], D))
            
            H_ab = get_intercomp_interactions_H(A, M, U, t_dict, statistic=stat, component_count=component_count)
            H_ab = sc.sparse.coo_matrix((H_ab[0], (H_ab[1], H_ab[2])), shape=(D, D))
            
            for delta_i in range(0, delta_vec.shape[0]):
                
                
                delta = delta_vec[delta_i]
                H_hop_1 = [List(), List(), List()]
                H_hop_1 = get_kinetic_H(
                    A,
                    M,
                    J,
                    t_dict,
                    delta_t=delta,
                    pbc=pbc,
                    statistic=stat[0],
                    component_count=component_count,
                    component_no=0,
                )
                H_hop_1 = sc.sparse.coo_matrix(
                    (H_hop_1[0], (H_hop_1[1], H_hop_1[2])), shape=(D, D)
                )
                H_hop_2 = get_kinetic_H(
                    A,
                    M,
                    J,
                    t_dict,
                    delta_t=delta,
                    pbc=pbc,
                    statistic=stat[1],
                    component_count=component_count,
                    component_no=1,
                )
                H_hop_2 = sc.sparse.coo_matrix(
                    (H_hop_2[0], (H_hop_2[1], H_hop_2[2])), shape=(D, D)
                )
                H_1 = J*H_hop_1 + V[0]*H_aa
                
                H_2 = J*H_hop_2 + V[1]*H_bb
            
                
                H_tot = H_1 + H_2 + U*H_ab
                
        
        
                # ax[num] = plot_coo_matrix(H_out)
                # sns.heatmap(np.matrix(H_out.toarray()), square=True, cmap=/'coolwarm', annot=False, ax=ax[num], cbar=False)
        
                # res_dict[t_var] = sc.sparse.linalg.eigsh(mat_res, k=100, return_eigenvectors=False, which='SM')
                evals_H1, evecs_H1 = np.linalg.eigh(H_1.toarray())
                evals_H2, evecs_H2 = np.linalg.eigh(H_2.toarray())
                evals, evecs = np.linalg.eigh(H_tot.toarray())
                # vals, vecs = sc.linalg.eigh(mat_res.toarray(), eigvals_only=False)
                idx = np.argsort(evals, -1)
                evecs = evecs[:, idx]
                evals = evals[idx]
                evecs_dt = []
                evals_dt = []
                for evals_i in range(0, evals.shape[0]):
                    psi = evecs[:, evals_i].copy()
                    psi_H1 = evecs_H1[:, evals_i].copy()
                    psi_H2 = evecs_H2[:, evals_i].copy()
                    e_i = evals[evals_i]
                    exp_val_H_1 = np.real(np.dot(np.conjugate(psi).T, np.dot(H_1.toarray(), psi)))
                    exp_val_H_2 = np.real(np.dot(np.conjugate(psi).T, np.dot(H_2.toarray(), psi)))
                    exp_val_H_tot = np.real(np.dot(np.conjugate(psi).T, np.dot(H_tot.toarray(), psi)))
                    Evals1_matrix[delta_i, evals_i] = exp_val_H_1
                    Evals2_matrix[delta_i, evals_i] = exp_val_H_2
                    Evals_tot_matrix[delta_i, evals_i] = exp_val_H_tot
                    # evecs_dt.append()
                    print("stat = " + str(stat) + ", U_aa = {:.1f}, U_bb = {:.1f}, U_ab = {:.1f}, ".format(V[0], V[1], U)+
                        "dt = {:.1f}, i = {:2d}, E_1 = {:.3f}, E_2 = {:.3f}, eval_i = {:.3f}".format(
                            delta, evals_i, exp_val_H_1, exp_val_H_2, exp_val_H_tot
                        )
                    )
                    
            figs, ax = plt.subplots(1, 3, figsize=(36, 14))
            figs.tight_layout(pad=16)
            figs.suptitle("stat="+str(stat)+" | N="+str(N)+" | J = {} |M={} | V_aa={:.2f} | V_bb = {:.2f} | U_ab = {:.2e} | pbc = {}".format(J, M, V[0], V[1], U, pbc)+
                          "\n"+r"$\hat{H}_{\sigma} = -J\sum_{i = 0}^{M-1} (1 + (-1)^{i+1} \Delta t)(a^\dagger_{\sigma, i} a_{\sigma, i+1} + h.c) + \frac{V_{\sigma\sigma}}{2}\sum_{i = 0}^{M} \hat{n}_{\sigma, i}(\hat{n}_{\sigma, i} - 1), \quad \hat{H}_{ab} = U_{ab}\sum_{i = 0}^{M} \hat{n}_{a, i}\hat{n}_{b, i}, \quad \sigma \in {a, b}$"+"\n"+r"$\hat{H}_{tot} = H_1 + H_2 + H_{ab}$" + r"$, \quad \hat{H}_{total} | \lambda \rangle = E_{\lambda} | \lambda \rangle$", fontsize=20)
            for i in range(0, Evals1_matrix.shape[1]):
                ax[0].set_title(r"$\langle \lambda | H_1 | \lambda \rangle$", fontsize=18)
                ax[0].set_xlabel(r"$\Delta t$", fontsize=16)
                ax[0].set_ylabel(r"$E = \langle \lambda | H_1 | \lambda \rangle$", fontsize=20)
                ax[0].plot(delta_vec, Evals1_matrix[:, i])
                ax[1].set_title(r"$\langle \lambda | H_2 | \lambda \rangle $", fontsize=18)
                ax[1].plot(delta_vec, Evals2_matrix[:, i])
                ax[1].set_xlabel(r"$\Delta t$", fontsize=16)
                ax[1].set_ylabel(r"$E = \langle \lambda | H_2 | \lambda \rangle$", fontsize=20)
                ax[2].set_title(r"$\langle \lambda | H_{total} | \lambda \rangle $", fontsize=18)
                ax[2].plot(delta_vec, Evals_tot_matrix[:, i])
                ax[2].set_xlabel(r"$\Delta t$", fontsize=16)
                ax[2].set_ylabel(r"$E_{\lambda}$", fontsize=22)
                
            # figs.savefig("Eigenenergies_Plots/EE_plot_stat="+str(stat)+"_|_N="+str(N)+"_|_J={}_|M={}_|_V_aa={:.2f}_|_V_bb={:.2f}_|_U_ab={:.2e}_|_pbc={}".format(J, M, V[0], V[1], U, pbc)+".pdf")