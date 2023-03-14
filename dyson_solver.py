import numpy as np
from pydlr import kernel, dlr
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

class Symmetrizer:

    def __init__(self, nx, no):
        self.N = (no*(no-1))//2
        self.nx, self.no = nx, no
        self.diag_idxs = np.arange(self.no)
        self.triu_idxs = np.triu_indices(no, k=1)
        self.tril_idxs = np.tril_indices(no, k=-1)
    
    def get_x_d(self, g_xaa):
        x_d = g_xaa[:, self.diag_idxs, self.diag_idxs].flatten()
        return x_d

    def set_x_d(self, g_xaa, x_d):
        g_xaa[:, self.diag_idxs, self.diag_idxs] = x_d.reshape((self.nx, self.no))
        return g_xaa

    def get_x_u(self, g_xaa):
        x_u = g_xaa[:, self.triu_idxs[0], self.triu_idxs[1]].flatten()
        return x_u

    def set_x_u(self, g_xaa, x_u):
        g_xaa[:, self.triu_idxs[0], self.triu_idxs[1]] = x_u.reshape((self.nx, self.N))
        g_xaa[:, self.tril_idxs[0], self.tril_idxs[1]] = g_xaa[:, self.triu_idxs[0], self.triu_idxs[1]].conj()
        #if complex: g_xaa += np.transpose(g_xaa, axes=(0,2,1)).conj()
        return g_xaa
    
    def get_x_l(self, g_xaa):
        x_l = g_xaa[:, self.tril_idxs[0], self.tril_idxs[1]].flatten()
        return x_l
    
    def set_x_l(self, g_xaa, x_l):
        g_xaa[:, self.tril_idxs[0], self.tril_idxs[1]] = x_l.reshape((self.nx, self.N))
        return g_xaa
        

    def get_diag_indices(self): return self.diag_idxs
    def get_triu_indices(self): return self.triu_idxs

def constrained_lstsq_dlr_from_tau(d,         # dlr class
                                   iw_or_tau, # iomega or tau mesh
                                   g_iaa,     # G data  on tau or iw 
                                   g0_iaa,    # G0 data on tau or iw
                                   beta,      # inverse temperature
                                   sigma_moments, # high-freq moments of Σ
                                   method = {'norm': 'tau',  # norm -> { 'tau', 'iw'}
                                             'optim': 'sigma', # optim -> { 'dlr', 'sigma'}
                                            }, # driver for different settings
                                   ftol=1e-9, 
                                   symmetrizer=None,
                                   verbose = True,
                                   minimal_dlr=False,
                                   initial_sigma=None
                                  ):
    
    
    
    
    
    nx = len(d)
    ni, no, _ = g_iaa.shape
    shape_xaa = (nx, no, no)
    N = (no*(no-1))//2

    dtype = complex
    nX = nx * (no + 2*N)
    
    # precompute Mkl
    def compute_mkl(d):
        Mkl = np.zeros((len(d), len(d)), dtype=np.float128)
        for iwk, wk in enumerate(d.dlrrf):
            for iwl, wl in enumerate(d.dlrrf):
                K0wk, Kbwk = kernel(np.array([0.,1.]), np.array([wk]))
                K0wl, Kbwl = kernel(np.array([0.,1.]), np.array([wl]))
                if np.fabs(wk+wl) < 1e-13:
                    Mkl[iwk,iwl] = K0wk*K0wl
                else:
                    Mkl[iwk, iwl] = (K0wk*K0wl - Kbwk*Kbwl)
                    Mkl[iwk, iwl] /= ((wk+wl))
        return Mkl
    
    # fold and unfold complex numbers
    
    def merge_re_im(x):
        x_d, x_u = x[:nx*no], x[nx*no:]
        re, im = np.split(x_u, 2)
        x_u = re + 1.j * im
        return x_d, x_u

    def split_re_im(x_d, x_u):
        return np.concatenate((
            np.array(x_d.real, dtype=float),
            np.array(x_u.real, dtype=float),
            np.array(x_u.imag, dtype=float)))
                                   
    # Greens function <-> vector conversion

    sym = symmetrizer if symmetrizer is not None else Symmetrizer(nx, no)
    
    def sig_from_x(x):
        x_d, x_off = x[:2*nx*no], x[2*nx*no:]
        x_u, x_l = np.split(x_off, 2)
    
        re, im = np.split(x_d, 2)
        x_d = re + 1.j * im
        
        re, im = np.split(x_u, 2)
        x_u = re + 1.j * im
        
        re, im = np.split(x_l, 2)
        x_l = re + 1.j * im
        
        sig = np.zeros((nx, no, no), dtype=dtype)
        sym.set_x_u(sig, x_u)
        sym.set_x_l(sig, x_l)
        sym.set_x_d(sig, x_d)
        
        return sig

    def x_from_sig(sig):
        x_d = sym.get_x_d(sig);
        x_u = sym.get_x_u(sig);
        x_l = sym.get_x_l(sig);
        x = np.concatenate((np.array(x_d.real, dtype=float),
                        np.array(x_d.imag, dtype=float),
                        np.array(x_u.real, dtype=float),
                        np.array(x_u.imag, dtype=float),
                        np.array(x_l.real, dtype=float),
                        np.array(x_l.imag, dtype=float)
                      ))
    
        return x
    
    # constraint
    sig_infty, sigma_1 = sigma_moments[0], sigma_moments[1]
        
    def mat_vec(mat):
        v_d = sym.get_x_d(mat[None, ...])
        v_u = sym.get_x_u(mat[None, ...])
        v_l = sym.get_x_l(mat[None, ...])
        
        return np.concatenate((
            np.array(v_d.real, dtype=float),
            np.array(v_d.imag, dtype=float),
            np.array(v_u.real, dtype=float),
            np.array(v_u.imag, dtype=float),
            np.array(v_l.real, dtype=float),
            np.array(v_l.imag, dtype=float)
        ))
        
    def constraint_func(x):
    
        if method['optim'] == 'sigma':
            sig = d.dlr_from_matsubara(sig_from_x(x), beta)
            mat = -sig.sum(axis=0)
        else:
            sig = sig_from_x(x)
            mat = -sig.sum(axis=0)
            
        vec = mat_vec(mat)
        return vec
    
    bound = mat_vec(sigma_1)
    
    constraints = (NonlinearConstraint(constraint_func,
                                           bound, bound))


    # target function   
    def dyson_difference(x):
        if method['optim'] == 'sigma':
            sig = sig_from_x(x)
        else:
            sig = d.matsubara_from_dlr(sig_from_x(x), beta)
        
        sig_iwaa = sig + sig_infty
            
        #  G - G0 - G0*Σ*G = 0 done on the DLR nodes
        r_iwaa = g_iwaa - g0_iwaa - g0_iwaa@sig_iwaa@g_iwaa
        
        if method['norm'] == 'tau':
            # compute DLR of rk_iwaa
            r_xaa = d.lstsq_dlr_from_matsubara(freq, r_iwaa, beta)
            # ||R||^2 = r^T @ M @ r
            R2 = np.einsum('mnk, kl, lnm->nm', r_xaa.T.conj(), Mkl, r_xaa).flatten()
            return np.sqrt(np.sum(R2)).real
        else:
            return np.linalg.norm(r_iwaa.flatten())
        
    def target_function(x):
        y = dyson_difference(x)
        return y 
        
    if method['norm'] == 'tau':
        # compute Mkl
        Mkl = compute_mkl(d)
        
    freq = d.get_matsubara_frequencies(beta)

    if minimal_dlr:
        triqs_freq = iw_or_tau 
        dlr_grid_indices=np.array([np.where(abs(triqs_freq-f) < 1e-8)[0][0] for f in freq])
        g_iwaa, g0_iwaa = g_iaa[dlr_grid_indices], g0_iaa[dlr_grid_indices]
    else:
        # dlr fit to G and G0
        g_xaa = d.lstsq_dlr_from_tau(iw_or_tau, g_iaa, beta)
        g0_xaa = d.lstsq_dlr_from_tau(iw_or_tau, g0_iaa, beta)
        # compute and obtain initial Σ
        g_iwaa = d.matsubara_from_dlr(g_xaa, beta)
        g0_iwaa = d.matsubara_from_dlr(g0_xaa, beta)
    
    if verbose and not minimal_dlr:
        g=d.eval_dlr_tau(g_xaa, iw_or_tau, beta)
        g0=d.eval_dlr_tau(g0_xaa, iw_or_tau,  beta)
        
        print('initial DLR fits to G(τ) and G0(τ)')
        print(f'|G(τ) - Gdlr(τ)| = {np.max(np.abs(g-g_iaa)):.6e}')
        print(f'|G0(τ) - G0dlr(τ)| = {np.max(np.abs(g0-g0_iaa)):.6e}')

    # the DLR representable part of the self-energy
    if initial_sigma is None:
        sig0_iwaa = np.linalg.inv(g0_iwaa)-np.linalg.inv(g_iwaa)-sig_infty
    else:
        sig0_iwaa = initial_sigma[dlr_grid_indices]-sig_infty
    
    if verbose:
        check = sig_from_x(x_from_sig(sig0_iwaa))
        assert np.allclose(check, sig0_iwaa), "sigma converter is broken!"
    
    if method['optim'] == 'sigma': 
        # optimize Σ(iν)
        x_init = x_from_sig(sig0_iwaa)
    else:
        # optimize σ -> Σ(iν) = Aσ + Σoo
        x_init = x_from_sig(d.dlr_from_matsubara(sig0_iwaa, beta))

    sol = minimize(target_function, 
                   x_init,
                   method='SLSQP', 
                   constraints=constraints,
                   options=dict(ftol=ftol, 
                                maxiter=10000, 
                                disp=True if verbose else False),
        )
    
    if verbose: print(sol.success, sol.message)
    if not sol.success: print('[WARNING] Minimization did not converge!')
    
    if method['optim'] == 'sigma':
        sig_iwaa = sig_from_x(sol.x)
        sig_xaa = d.dlr_from_matsubara(sig_iwaa, beta)
    else:
        sig_xaa = sig_from_x(sol.x)
        
    if verbose: print(f'Σ1 constraint diff: {np.max(np.abs(-sig_xaa.sum(axis=0)-sigma_1)):.4e}')

    return sig_xaa, sol

def dyson_solver(d, Sigma_iw_raw, G_tau_or_iw, G0_tau_or_iw, Sigma_moments, minimal_dlr=False, initial_sigma=None):

    beta = G_tau_or_iw.mesh.beta

    Sigma_iw_fit = Sigma_iw_raw.copy()
    mesh = np.array([float(x) for x in G_tau.mesh]) if not minimal_dlr else np.array([complex(x) for x in Sigma_iw_fit.mesh])
    
    for block, sig in Sigma_iw_fit:
        sig_xaa, sol = constrained_lstsq_dlr_from_tau(d, 
                                                      mesh, 
                                                      G_tau_or_iw[block].data,
                                                      G0_tau_or_iw[block].data, 
                                                      beta,
                                                      Sigma_moments[block],
                                                      minimal_dlr=minimal_dlr,
                                                      initial_sigma=initial_sigma[block].data if initial_sigma is not None else None
                                                      )

        Sigma_iw_fit[block].data[:] = d.eval_dlr_freq(sig_xaa, mesh, beta)
        Sigma_iw_fit[block].data[:] += Sigma_moments[block][0]

    return Sigma_iw_fit
