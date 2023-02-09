from pydlr import dlr, kernel
from h5 import HDFArchive
from triqs.gf import *
import numpy as np 
from triqs.atom_diag import trace_rho_op
from triqs.operators import *
import scipy.optimize as optimize
from scipy import linalg


def fit_dlr(gf, **kwargs):
    """
    Obtain Discrete Lehemann Representation (DLR) of imaginary time Green's function.
    
    This is a wrapper to the libdlr library: https://github.com/jasonkaye/libdlr
    """
    from pydlr import dlr, kernel

    if isinstance(gf, BlockGf):
        gf_ = gf.copy()
        for block, g in gf_:
            g.data[:,:,:] = fit_dlr(g, **kwargs)
        return gf_

    is_gf = isinstance(gf, Gf)
    is_mesh = (isinstance(gf.mesh, MeshImTime) or isinstance(gf.mesh, MeshImFreq))

    assert is_gf and is_mesh, "fit_dlr expects imaginary time or Matsubara Green's function objects."

    # construct DLR basis
    dlr_basis = dlr(**kwargs)

    if isinstance(gf.mesh, MeshImTime):
        tau = np.array([x.real for x in gf.mesh], dtype=float)
        beta = gf.mesh.beta
        Gdlr = dlr_basis.lstsq_dlr_from_tau(tau, gf.data, beta)
        return dlr_basis.eval_dlr_tau(Gdlr, tau, beta)

    if isinstance(gf.mesh, MeshImFreq):
        iwn= np.array([x.real + 1j*x.imag for x in gf.mesh], dtype=complex)
        beta = gf.mesh.beta
        Gdlr = dlr_basis.lstsq_dlr_from_matsubara(iwn, gf.data, beta)
        return dlr_basis.eval_dlr_freq(Gdlr, iwn, beta)

    
def fit_dlr_mod(gf, constraints, **kwargs):
    """
    Obtain Discrete Lehemann Representation (DLR) of imaginary time Green's function.
    
    This is a wrapper to the libdlr library: https://github.com/jasonkaye/libdlr
    """
    from pydlr import dlr, kernel

    if isinstance(gf, BlockGf):
        
        gf_ = gf.copy()
        for block, g in gf_:
            d = constraints[block][2]
            g.data[:,:,:] = fit_dlr_mod(g, d, **kwargs)
        return gf_

    is_gf = isinstance(gf, Gf)
    is_mesh = (isinstance(gf.mesh, MeshImTime) or isinstance(gf.mesh, MeshImFreq))

    assert is_gf and is_mesh, "fit_dlr expects imaginary time or Matsubara Green's function objects."

    # construct DLR basis
    dlr_basis = dlr(**kwargs)
    

    if isinstance(gf.mesh, MeshImTime):
        tau = np.array([x.real for x in gf.mesh], dtype=float)
        beta = gf.mesh.beta
        Gdlr = dlr_basis.lstsq_dlr_from_tau(tau, gf.data, beta)
        return dlr_basis.eval_dlr_tau(Gdlr, tau, beta)

    if isinstance(gf.mesh, MeshImFreq):
        
        
        def fmin(x, A, b):
            y = np.dot(A,x) - b
            return np.dot(y,y).real
        
        iwn= np.array([x.real + 1j*x.imag for x in gf.mesh], dtype=complex)
        beta = gf.mesh.beta
        x0 = dlr_basis.lstsq_dlr_from_matsubara(iwn, gf.data, beta)
        x  = np.empty_like(x0)
        A = -1./(iwn[:,None]-dlr_basis.dlrrf[None, :]/beta)
        for orb1 in range(x0.shape[-1]):
            for orb2 in range(x0.shape[-1]):
                b = gf.data[:,orb1,orb2]
                G1 = constraints[orb1,orb2]

                cons   = [{'type': 'eq', 'fun': lambda x : 1-np.sum(x).real}, 
                          {'type': 'eq', 'fun': lambda x : G1.real - np.dot(dlr_basis.dlrrf/beta,x).real}
                         ]
                res = optimize.minimize(fmin, x0[:,orb1,orb2], (A, b), constraints=cons).x
       
                x[:,orb1,orb2] = res
        
        return dlr_basis.eval_dlr_freq(x, iwn, beta)


def fit_and_fourier_dlr(Gtau, n_points, **kwargs):
    
    from pydlr import dlr, kernel
    
    # G(τ) (QMC) - G (DLR) -> G(iω)
    assert isinstance(Gtau.mesh, MeshImTime)
    
    dlr_basis = dlr(**kwargs)
    
    tau = np.array([x.real for x in Gtau.mesh], dtype=np.float32)
    beta = Gtau.mesh.beta
    
    name_list = [name for name, g in Gtau]
    glist = [GfImFreq(indices=g.indices, beta=beta,n_points=n_points) for _, g in Gtau]
    
    Giw = BlockGf(name_list=name_list, block_list=glist, make_copies=True)
    iwn = np.array([x.real+1j*x.imag for x in Giw.mesh])
    for block, g in Gtau:
        Gdlr = dlr_basis.lstsq_dlr_from_tau(tau, g.data, beta)
        Giw[block].data[:,:,:] = dlr_basis.eval_dlr_freq(Gdlr, iwn, beta)
    return Giw


def fit_and_fourier_dlr_mod(Gtau, n_points, constrain=False, **kwargs):
    #print(constraints)
    
    from pydlr import dlr, kernel
    
    # G(τ) (QMC) - G (DLR) -> G(iω)
    assert isinstance(Gtau.mesh, MeshImTime)
    
    dlr_basis = dlr(**kwargs)
        
    tau = np.array([x.real for x in Gtau.mesh])
    beta = Gtau.mesh.beta
    
    name_list = [name for name, g in Gtau]
    glist = [GfImFreq(indices=g.indices, beta=beta,n_points=n_points) for _, g in Gtau]
    
    Giw = BlockGf(name_list=name_list, block_list=glist, make_copies=True)
    
    iwn = np.array([x.real+1j*x.imag for x in Giw.mesh])
    
    if constrain:
        A = kernel(tau/beta, dlr_basis.dlrrf)
        for block, g in Gtau:
            x0 = dlr_basis.lstsq_dlr_from_tau(tau, g.data, beta)
            x = np.empty_like(x0)
            for orb1 in range(x0.shape[-1]):
                for orb2 in range(x0.shape[-1]):
                    
                    C = g.data[:,orb1,orb2]
                    C = C.reshape(C.shape[0], 1)
                    
                    if orb1 == orb2:
    
                        B = np.zeros((1, A.shape[1]));
                        B[0, :] -= 1.0
    
                        D = np.zeros((1,1))
                        D[0, 0] = 1.0
                        x[:,orb1,orb2] = linalg.lapack.zgglse(A, B, C.real, D)[3]
                    else:
                        
                        x[:,orb1,orb2] = np.linalg.lstsq(A, C.real)[0].flatten()
                        
            Giw[block].data[:,:,:] = dlr_basis.eval_dlr_freq(x, iwn, beta)
            
    else:
        for block, g in Gtau:
            Gdlr = dlr_basis.lstsq_dlr_from_tau(tau, g.data, beta)
            Giw[block].data[:,:,:] = dlr_basis.eval_dlr_freq(Gdlr, iwn, beta)
    return Giw

def make_noisy_tau(G_tau, noise):
    G_qmc = G_tau.copy()
    for name, g in G_qmc:
        shape = g.data.shape
        g.data[:,:,:] +=  np.random.normal(scale=noise, size=shape)
    return G_qmc


def fit_dlr_sigma(Sigma, constraints, **kwargs):
    
    
    from pydlr import dlr, kernel
    

    dlr_basis = dlr(**kwargs)

    Sigma_dlr = Sigma.copy()
    
    iwn = np.array([x.real+1j*x.imag for x in Sigma_dlr.mesh])
    beta = Sigma_dlr.mesh.beta
    
    A = -1.0/(iwn[:,None] - dlr_basis.dlrrf[None, :]/beta)
    
    for block, g in Sigma_dlr:
        
        wk, orb = A.shape[-1], g.data.shape[-1]
        x = np.zeros((wk, orb, orb), dtype=complex)
        for orb1 in range(orb):
            for orb2 in range(orb):
                C = g.data[:, orb1, orb2] - constraints[block][0][orb1,orb2] # use Sigma_infty
                C = C.reshape(C.shape[0], 1)
                B = np.zeros((1, A.shape[1]),dtype=complex)
                B[0,:] -= 1.0
                D = np.zeros((1,1),dtype=complex)
                D[0,0] = constraints[block][1][orb1,orb2]
            
                x[:,orb1,orb2] = linalg.lapack.zgglse(A, B, C, D)[3]

        Sigma_dlr[block].data[:,:,:] = dlr_basis.eval_dlr_freq(x, iwn, beta) + constraints[block][0][orb1,orb2]
    return Sigma_dlr


def anticomm(A,B): return A*B + B*A
def comm(A,B): return A*B - B*A

def green_high_frequency_moments(density_matrix,
                           ad_imp, 
                           gf_struct, 
                           h_imp):
    """
    Calculate the first and second high frequency moment of G_iw
    following Rev. Mod. Phys. 83, 349 (2011). They read
    (0) G_0           =    0
    (1) G_1           =  <{c,c+}>
    (2) G_2           = -<{[H,c],c+}>
    where H is the impurity Hamiltonian (H = impurity levels + Hint).
    Parameters
    ----------
    density_matrix : list, np.ndarray
                     measured density matrix from TRIQS/CTHYB.
    ad_imp         : AtomDiag
                     h_loc_diagonalization from TRIQS/CTHYB.
    gf_struct      : List of pairs (str,int)
                     Block structure of Green's function.
    h_imp          : triqs.operators.Operator
                     impurity Hamiltonian   
    Returns
    -------
    green_moments  : dict, np.ndarray
                     first and second moments in a dict with the
                     same block strucutre of the TRIQS Gf object.
    """

    green_moments = {bl : np.zeros((3, bl_size, bl_size),dtype=complex) for bl, bl_size in gf_struct}
    for bl, bl_size in gf_struct:
        # G_0/iwn = 1/iwn
        green_moments[bl][1] = np.eye(bl_size)
        for orb1 in range(bl_size):
            for orb2 in range(bl_size):
                # G_1/iwn**2 term
                op = -anticomm(comm(h_imp, c(bl,orb1)), c_dag(bl,orb2))
                green_moments[bl][2,orb1,orb2] = trace_rho_op(density_matrix, op, ad_imp)

    return green_moments


def sigma_high_frequency_moments(density_matrix,
                           ad_imp, 
                           gf_struct, 
                           h_int):
    """
    Calculate the first and second high frequency moment of Sigma_iw
    following Rev. Mod. Phys. 83, 349 (2011). They read
    (0) Sigma_0       = -<{[Hint,c],c+}> (Hartree shift)
    (1) Sigma_1       =  <{[Hint,[Hint,c]],c+}> - Sigma_0^2,
    where Hint is the interaction Hamiltonian
    Parameters
    ----------
    density_matrix : list, np.ndarray
                     measured density matrix from TRIQS/CTHYB.
    ad_imp         : AtomDiag
                     h_loc_diagonalization from TRIQS/CTHYB.
    gf_struct      : List of pairs (str,int)
                     Block structure of Green's function.
    h_int          : triqs.operators.Operator
                     interaction Hamiltonian
    Returns
    -------
    sigma_moments  : dict, np.ndarray
                     first and second moments in a dict with the
                     same block strucutre of the TRIQS Gf object.
    """


    sigma_moments = {bl : np.zeros((2, bl_size, bl_size),dtype=complex) for bl, bl_size in gf_struct}
    for bl, bl_size in gf_struct:
        for orb1 in range(bl_size):
            for orb2 in range(bl_size):

                # Sigma_HF term
                op_HF = -anticomm(comm(h_int, c(bl,orb1)), c_dag(bl,orb2))
                sigma_moments[bl][0,orb1,orb2] = trace_rho_op(density_matrix, op_HF, ad_imp)

                # Sigma_1/iwn term
                op_iw = anticomm(comm(h_int, comm(h_int, c(bl,orb1))), c_dag(bl,orb2))
                sigma_moments[bl][1,orb1,orb2] = trace_rho_op(density_matrix, op_iw, ad_imp) - sigma_moments[bl][0,orb1,orb2]**2

    return sigma_moments


# use the G_moments to the Fourier transform of Gtau
def fourier_with_moments(Gtau, n_points, known_moments):
    beta = Gtau.mesh.beta
    
    name_list = [name for name, g in Gtau]
    glist = [GfImFreq(indices=g.indices, beta=beta,n_points=n_points) for _, g in Gtau]
    
    Giw = BlockGf(name_list=name_list, block_list=glist, make_copies=True)
    
    for bl, g in Gtau:
        bl_size = g.target_shape[0]
        Giw[bl].set_from_fourier(g, known_moments[bl])
    return Giw

def repair_tail(Sigma, moments):
    
    Sigma_repair = Sigma.copy()
    
    iwn = np.array([x.real+1j*x.imag for x in Sigma_repair.mesh], dtype=complex)
    
    pos_freq = np.where(iwn.imag > 0)
    
    mid = len(iwn)//2
    
    for name, sig in Sigma_repair:
        bl_size = sig.target_shape[0]
        for orb1 in range(bl_size):
            for orb2 in range(bl_size):
                Sigma_asymp = moments[name][0][orb1,orb2]+moments[name][1][orb1,orb2]/iwn
                
                min_func = np.abs(sig.data[:,orb1,orb2].imag-Sigma_asymp.imag)
                replace = mid + np.argmin(min_func[pos_freq])
                
                #if orb1 == orb2: print("replacing tail at iωn = ", iwn[replace].imag)
                for i in range(replace,len(iwn)): Sigma_repair[name].data[i,orb1,orb2] = Sigma_asymp[i]
                for i in range(len(iwn)-replace): Sigma_repair[name].data[i,orb1,orb2] = Sigma_asymp[i]
    return Sigma_repair

ar = HDFArchive('cthyb-pm.h5')
dm = ar['dm']
hdiag = ar['h_loc_diag']
G_iw_data = ar['G']
G_tau_data = ar['Gtau']
del ar
