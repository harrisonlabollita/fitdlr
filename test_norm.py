#!/usr/bin/env python3
import numpy as np
from triqs.gf import *
from pydlr import kernel, dlr

beta = 10
eps  = 1
analytic_norm = np.sqrt(np.tanh(0.5*beta*eps)/(2*eps*beta))
# G(τ) = exp(-τϵ)/( 1 + exp(-βϵ))

tau_mesh = MeshImTime(beta=beta, S='Fermion', n_tau=10001)
tau_i = np.array([float(x) for x in tau_mesh])
G = Gf(mesh=tau_mesh, target_shape=[1,1])
for tau in G.mesh: G[tau] = -np.exp((beta*(eps<0) - tau.value) * eps) / (1. + np.exp(-beta * abs(eps)))

d = dlr(lamb=30, eps=1e-9)
g_xaa = d.lstsq_dlr_from_tau(tau_i, G.data, beta)
Mkl = np.zeros((len(d), len(d)),dtype=np.float128)
for iwk, wk in enumerate(d.dlrrf):
    for iwl, wl in enumerate(d.dlrrf):
        K0wk, Kbwk = kernel(np.array([0.,1.]), np.array([wk])).flatten()
        K0wl, Kbwl = kernel(np.array([0.,1.]), np.array([wl])).flatten()
        if np.fabs(wk+wl) < 1e-13:
            Mkl[iwk,iwl] = K0wk*K0wl
        else:
            Mkl[iwk, iwl] = (K0wk*K0wl - Kbwk*Kbwl)
            Mkl[iwk, iwl] /= ((wk+wl))
dlr_norm = np.sqrt(g_xaa.flatten().T@Mkl@g_xaa.flatten())

print("||G|| (ana) = ", analytic_norm)
print('||G|| (dlr) = ', dlr_norm)
print(f'Δ||G|| = {abs(analytic_norm-dlr_norm.real):.10e}')
