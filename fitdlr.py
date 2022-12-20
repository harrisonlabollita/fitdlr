""" 
Imaginary time Green's function fit routines using the
Discrete Lehmann Representation (DLR).

"""

import numpy as np

from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from pydlr.kernel import kernel


def constrained_lstsq_dlr_from_tau(
        dlr, h_ab, U_abcd, tau_i, g_iaa, beta,
        positivity=True, discontinuity=True, density=True, realvalued=False,
        ftol=1e-9, symmetrizer=None):
    
    """ DLR imaginary time Green's function fit with constraints.

    - Imposing the (fermionic) discontinuity boundary condition.
    - Consistency between the density matrix and the static self-energy

    Author: Hugo U.R. Strand (2022) 

    Parameters
    ----------

    dlr : dlr
        Discrete Lehmann Representation object from pydlr
    h_ab : (n,n) array_like 
        Static quadratic term in Green's function
    U_abcd : (n,n,n,n) array_like
        Interaction tensor for the system
    tau_i : (nt) array_like
        Imaginary time points of input Green's function
    g_iaa : (nt,n,n) array_like
        Imaginary time Green's function data
    beta : float
        Inverse temperature

    positivity : bool, optional
        Impose spectral positivity of diagonal Green's function components. Default `True`.
    discontinuity : bool, optional
        Impose (fermionic) discontinuity boundary condition. Default `True`.
    density: bool, optional
        Impose consistency between the density matrix and the static self-energy. Default `True`.
    realvalued : bool, optional
        Impose real-valued symmetric Green's function
    ftol : float, optional
        Precision goal for the fitting.

    Returns
    -------

    g_xaa : (len(dlr),n,n) array_like
        Fitted DLR coefficients for the Green's function
    sol : scipy.OptimizeResult
        Instance with minimization information
    """

    np.testing.assert_array_almost_equal(h_ab, h_ab.T.conj())
    
    nx = len(dlr)
    ni, no, _ = g_iaa.shape
    shape_xaa = (nx, no, no)
    N = (no*(no-1))//2

    # -- Real/Hermitian setup
    
    if realvalued:
        dtype = float
        g_iaa = np.array(g_iaa.real, dtype=dtype)
        nX = nx * (no + N)
        #merge_re_im = lambda x : x[:nx*no], x[nx*no:]
        def merge_re_im(x):
            return x[:nx*no], x[nx*no:]
        split_re_im = lambda x_d, x_u : np.concatenate((
            np.array(x_d.real, dtype=float), np.array(x_u.real, dtype=float)))
    else:
        dtype = complex
        nX = nx * (no + 2*N)

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
                                   
    # -- Greens function <-> vector conversion

    sym = symmetrizer if symmetrizer is not None else  Symmetrizer(nx, no)
    
    def g_from_x(x):
        x_d, x_u = merge_re_im(x)
        g_xaa = np.zeros((nx, no, no), dtype=dtype)
        sym.set_x_u(g_xaa, x_u)
        sym.set_x_d(g_xaa, x_d)
        return g_xaa
                        
    def x_from_g(g_xaa):
        x_d = sym.get_x_d(g_xaa)
        x_u = sym.get_x_u(g_xaa)
        x = split_re_im(x_d, x_u)
        return x

    # -- Setup constraints
    
    constraints = []

    if positivity:
        #for i in range(no):
        for i in sym.get_diag_indices():
            A_xaa = np.zeros(shape_xaa, dtype=dtype)
            A_xaa[:, i, i] = 1.
            A_nX = x_from_g(A_xaa)[None, :]

            #if np.max(A_nX) == 0: continue
            
            disc_constr = LinearConstraint(A_nX, -float('inf'), 0.)
            constraints.append(disc_constr)
        
    if discontinuity:
        for i in sym.get_diag_indices():
            A_xaa = np.zeros(shape_xaa, dtype=dtype)
            A_xaa[:, i, i] = 1.
            A_nX = x_from_g(A_xaa)[None, :]
            bound = -1.
            disc_constr = LinearConstraint(A_nX, bound, bound)
            constraints.append(disc_constr)

        for i, j in zip(*sym.get_triu_indices()):
            A_xaa = np.zeros(shape_xaa, dtype=dtype)
            A_xaa[:, i, j] = 1.
            A_nX = x_from_g(A_xaa)[None, :]            

            bound = 0.
            disc_constr = LinearConstraint(A_nX, bound, bound)
            constraints.append(disc_constr)

            if not realvalued:
                A_xaa[:, i, j] = 1.j
                A_nX = x_from_g(A_xaa)[None, :]
                bound = 0.
                disc_constr = LinearConstraint(A_nX, bound, bound)
                constraints.append(disc_constr)
            
        #for i, j in zip(*np.triu_indices(no)):
        if False:
            A_xaa = np.zeros(shape_xaa, dtype=dtype)
            A_xaa[:, i, j] = 1.
            A_nX = x_from_g(A_xaa)[None, :]
            
            bound = -1. * (i == j)
            disc_constr = LinearConstraint(A_nX, bound, bound)
            constraints.append(disc_constr)

            if i != j and not realvalued:
                A_xaa[:, i, j] = 1.j
                A_nX = x_from_g(A_xaa)[None, :]
                bound = 0.
                disc_constr = LinearConstraint(A_nX, bound, bound)
                constraints.append(disc_constr)
                
    if density:
        rho_x = -kernel(np.array([1.]), dlr.dlrrf)[0]
        U_ab_xcd = -np.swapaxes(U_abcd, 0, 1)[:, :, None, :, :] * rho_x[None, None, :, None, None]
        U_nX = U_ab_xcd.reshape((no*no, nx*no*no))

        dK_0_x, dK_beta_x = kernel(np.array([0., 1.]), dlr.dlrrf) * (-dlr.dlrrf[None, :]/beta)
        dK_x = dK_0_x + dK_beta_x
        I = np.eye(no*no)
        dK_nX = np.einsum('x,AB->AxB', dK_x, I).reshape((no*no, nx*no*no))

        A_nX = dK_nX - U_nX

        def mat_vec(mat):
            v_d = sym.get_x_d(mat[None, ...]).real
            v_u = sym.get_x_u(mat[None, ...])
            v = split_re_im(v_d, v_u)
            return v
            
        def density_constraint_function(x):
            g = g_from_x(x)
            mat = A_nX.dot(g.flatten()).reshape(no, no)
            vec = mat_vec(mat)
            return vec
            
        density_bound = mat_vec(h_ab)
        density_constr = NonlinearConstraint(
            density_constraint_function, density_bound, density_bound)
        constraints.append(density_constr)

    def greens_function_difference(x):
        g_xaa = g_from_x(x)
        g_in_iaa = dlr.eval_dlr_tau(g_xaa, tau_i, beta) # straight up least squares
        err_iaa = g_in_iaa - g_iaa
        return err_iaa.flatten()
        
    def target_function(x):
        y = greens_function_difference(x)
        return np.linalg.norm(y)
    
    g0_xaa = dlr.lstsq_dlr_from_tau(tau_i, g_iaa, beta)
    g0_laa = dlr.tau_from_dlr(g0_xaa)
    x0 = x_from_g(g0_xaa)
    
    sol = minimize(
        target_function, x0,
        method='SLSQP', constraints=constraints,
        options=dict(ftol=ftol),
        )

    sol.res = np.max(np.abs(greens_function_difference(sol.x)))
    sol.g_xaa = g_from_x(sol.x)
    sol.norm = -np.sum(sol.g_xaa, axis=0)
    sol.norm_res = np.max(np.abs(sol.norm - np.eye(no)))

    sol.density_res = float('nan')
    if density: sol.density_res = np.max(np.abs(density_bound - density_constraint_function(sol.x)))

    return sol.g_xaa, sol


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
        #g_xaa += np.transpose(g_xaa, axes=(0,2,1)).conj()
        return g_xaa

    def get_diag_indices(self): return self.diag_idxs
    def get_triu_indices(self): return self.triu_idxs


def unique_non_zero(arr):
    return [ i for i in np.unique(arr) if i != 0 ]


class BlockSymmetrizer:

    def __init__(self, nx, block_mat):

        assert(len(block_mat.shape) == 2)
        assert(block_mat.shape[0] == block_mat.shape[1])

        #print(f'block_mat =\n{block_mat}')
        
        no = block_mat.shape[0]
        self.N = (no*(no-1))//2
        self.nx, self.no = nx, no
        
        term_idxs = unique_non_zero(block_mat)
        #print(f'term_idxs = {term_idxs}')

        diag_idxs = np.arange(no)
        diag_terms = unique_non_zero(block_mat[diag_idxs, diag_idxs])
        #print(f'diag_terms = {diag_terms}')

        triu_idxs = np.triu_indices(no, k=+1)
        triu_terms = unique_non_zero(block_mat[triu_idxs[0], triu_idxs[1]])
        #print(f'triu_terms = {triu_terms}')

        diag_term_idxs = []
        for diag_term in diag_terms:
            idxs = []
            for i in range(no):
                if block_mat[i, i] == diag_term:
                    idxs.append(i)
            diag_term_idxs.append(idxs)
            #print(diag_term, idxs)

        #print(f'diag_term_idxs = {diag_term_idxs}')

        triu_term_idxs = []
        for triu_term in triu_terms:
            idxs = []
            for i, j in zip(triu_idxs[0], triu_idxs[1]):
                if block_mat[i, j] == triu_term:
                    idxs.append((i, j))
            triu_term_idxs.append(idxs)
            print(triu_term, idxs)

        #print(f'triu_term_idxs = {triu_term_idxs}')

        self.diag_term_idx = [ idxs[0] for idxs in diag_term_idxs ]
        self.triu_term_idx = list(zip(*[ idxs[0] for idxs in triu_term_idxs ]))

        triu_term_idxs = [ list(zip(*term)) for term in triu_term_idxs ]
        #print(f'triu_term_idxs = {triu_term_idxs}')
        
        self.diag_terms, self.diag_term_idxs = diag_terms, diag_term_idxs
        self.triu_terms, self.triu_term_idxs = triu_terms, triu_term_idxs

        #print(f'diag_term_idx = {self.diag_term_idx}')
        #print(f'triu_term_idx = {self.triu_term_idx}')

        if len(self.triu_term_idx) == 0:
            self.triu_term_idx = [[], []]

    def get_x_d(self, g_xaa):
        x_d = g_xaa[:, self.diag_term_idx, self.diag_term_idx].flatten()
        return x_d

    def set_x_d(self, g_xaa, x_d):
        nx = g_xaa.shape[0]
        no = len(x_d) // nx
        for term_idx, diag_idxs in enumerate(self.diag_term_idxs):
            g_xaa[:, diag_idxs, diag_idxs] = x_d.reshape((nx, no))[:, term_idx][:, None]
        return g_xaa
        
    def get_x_u(self, g_xaa):
        x_u = g_xaa[:, self.triu_term_idx[0], self.triu_term_idx[1]].flatten()
        return x_u

    def set_x_u(self, g_xaa, x_u):
        for term_idx, triu_idxs in enumerate(self.triu_term_idxs):
            g_xaa[:, triu_idxs[0], triu_idxs[1]] = x_u.reshape((self.nx, self.N))[:, term_idx][:, None]

        g_xaa += np.transpose(g_xaa, axes=(0,2,1)).conj()
        return g_xaa
        
    def get_diag_indices(self): return self.diag_term_idx
    def get_triu_indices(self): return self.triu_term_idx
