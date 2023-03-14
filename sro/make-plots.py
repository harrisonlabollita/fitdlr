from triqs.gf import *
from triqs.atom_diag import trace_rho_op
from triqs.operators import n
from h5 import HDFArchive

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def max_G_diff(G1, G2, norm_temp = True):
    """
    calculates difference between two block Gfs
    uses numpy linalg norm on the last two indices first
    and then the norm along the mesh axis. The result is divided
    by sqrt(beta) for MeshImFreq and by sqrt(beta/#taupoints) for
    MeshImTime.
    1/ (2* sqrt(beta)) sqrt( sum_n sum_ij [abs(G1 - G2)_ij(w_n)]^2 )
    this is only done for MeshImFreq Gf objects, for all other
    meshes the weights are set to 1
    Parameters
    ----------
    G1 : Gf or BlockGf to compare
    G2 : Gf or BlockGf to compare
    norm_temp: bool, default = True
       divide by an additional sqrt(beta) to account for temperature scaling
       only correct for uniformly distributed error.
    __Returns:__
    diff : float
           difference between the two Gfs
    """

    if isinstance(G1, BlockGf):
        diff = 0.0
        for block, gf in G1:
            diff += max_G_diff(G1[block], G2[block], norm_temp)
        return diff

    assert G1.mesh == G2.mesh, 'mesh of two input Gfs does not match'
    assert G1.target_shape == G2.target_shape, 'can only compare Gfs with same shape'

    # subtract largest real value to make sure that G1-G2 falls off to 0
    if type(G1.mesh) is MeshImFreq:
        offset = np.diag(np.diag(G1.data[-1,:,:].real - G2.data[-1,:,:].real))
    else:
        offset = 0.0

    #  calculate norm over all axis but the first one which are frequencies
    norm_grid = abs(np.linalg.norm(G1.data - G2.data - offset, axis=tuple(range(1, G1.data.ndim))))
    # now calculate Frobenius norm over grid points
    norm = np.linalg.norm(norm_grid, axis=0)

    if type(G1.mesh) is MeshImFreq:
        norm = np.linalg.norm(norm_grid, axis=0) / np.sqrt(G1.mesh.beta)
    elif type(G1.mesh) is MeshImTime:
        norm = np.linalg.norm(norm_grid, axis=0) * np.sqrt(G1.mesh.beta/len(G1.mesh))
    elif type(G1.mesh) is MeshReFreq:
        norm = np.linalg.norm(norm_grid, axis=0) / np.sqrt(len(G1.mesh))
    else:
        raise ValueError('MeshReTime is not implemented')

    if type(G1.mesh) in (MeshImFreq, MeshImTime) and norm_temp:
        norm = norm / np.sqrt(G1.mesh.beta)

    return norm

def plot_self_energies(data):
	fig,ax = plt.subplots(len(data), 3,figsize=(12,9))
	plt.subplots_adjust(hspace=0.5)
	for ikey, key in enumerate(data.keys()):
		S_iw = data[key]['dmft_results/last_iter']['Sigma_iw']
		iw = np.array([x.real+1j*x.imag for x in S_iw.mesh])
		for i in range(3):
			orb = str(i)*2
			ax[ikey, i].plot(iw.imag, S_iw['up'].data[:,i,i].imag, '-', lw=2,label=r'Im$\Sigma^{'+orb+'}$'+' ($\Lambda = $'+str(key)+')')
			ax[ikey, i].set_ylim(-0.6,0.2)
			ax[ikey, i].axhline(0.0, color='k', ls='dotted')
			ax[ikey, i].legend(frameon=True, facecolor='white', edgecolor='white')
			ax[ikey, i].set_xlabel(r'$i\omega_{n}$'); #ax[i].set_ylabel('Im$\Sigma(i\omega_{n})$')
			ax[ikey, i].set_xlim(0, 60); 
#for a, let in zip(ax, ['(a)', '(b)', '(c)']):
#    a.text(-0.125, 1.125, let, transform = a.transAxes, size=20)
	plt.show()


if __name__ == "__main__":

    dlr_data = { 50: HDFArchive('sro_dmft_dlr_50_1e-10.h5'), 
                 60: HDFArchive('sro_dmft_dlr_60_1e-09.h5')
               }
    plot_self_energies(dlr_data)
