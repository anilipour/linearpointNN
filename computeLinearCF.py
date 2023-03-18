from classy import Class
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import mcfit
from astropy.table import Table
from astropy.io import ascii
import os

def cosmology(omega_m, omega_b, h, n_s, sigma_8, k_max = 1e3, zmax=100.):
    # set cosmological parameters
    H0 = h*100
    params = {'output' : 'mPk',
             'omega_b' : omega_b*h**2,
             'omega_cdm' : (omega_m - omega_b)*h**2,
             'H0' : H0,
             'n_s' : n_s,
             'sigma8' : sigma_8,
             'N_eff' : 3.046,
             'P_k_max_1/Mpc' : k_max,
             'z_max_pk' : zmax,
             'YHe' : 0.24672}
    
    # compute cosmology
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    return cosmo
    
def corr(cosmo, k_min, k_max, sampling = 100000, z = 0.0):
    # power spectrum sampling
    klin = np.logspace(np.log10(k_min), np.log10(k_max), num=sampling) # 1/MPc^-1    
    
    # linearly evolved power spectrum
    Pklin = np.array([cosmo.pk_lin(ki, z) for ki in klin])
    
    # klin in units h/Mpc
    # power in units (Mpc/h)^3
    h = cosmo.h()
    klin /= h
    Pklin *= h**3
    
    # correlation function
    CF = mcfit.P2xi(klin, l=0, lowring=True)
    r_lin, xi_lin = CF(Pklin, extrap=True)
    
    cf_table = Table([r_lin, xi_lin], names=('r', 'xi')) # astropy table

    return cf_table
    
    
def scales(cf_table):
    # spline fit
    r = cf_table['r']
    xi = cf_table['xi']
    spl = UnivariateSpline(r, xi, k=4, s=0)
    
    # critical points of correlation function
    deriv = spl.derivative()
    cps = deriv.roots() # roots only works on cubic spline, so original spline must be degree 4

    # get dip, peak, linear point
    dip = cps[0] # assuming the first critical point is the dip
    peak = cps[1] # and the second is the peak
    lp = (dip+peak)/2.0
    
    # get inflection point
    # spline fit of degree 5 to curve between dip and peak
    mask = (r < peak) & (r > dip)
    rMasked = r[mask]
    xiMasked = xi[mask]
    
    spl2 = UnivariateSpline(rMasked, xiMasked, k=5, s=0)
    deriv2 = spl2.derivative(2)
    ip = deriv2.roots()[0]
    
    return dip, peak, lp, ip

if __name__ == '__main__':
    path = '/home/ann22/project/bao_sims' # change to path with latin hypercube parameters file 
    os.chdir(path)
    
    simulation = 'latinHypercubeParameters.txt'
    param_list = []
    for line in open(simulation, 'r'): # get each line
        item = line.rstrip() # strip off newline and any other trailing whitespace
        param_list.append(item)

    del param_list[0]
    
    omegaMlist, omegaBlist, hList, nsList, s8list = [], [], [], [], []
    for item in param_list: # get radius and correlation from each line
        omegaM, omegaB, h, ns, s8 = item.split() # each line has both radius and correlation, so split
        omegaMlist.append(float(omegaM))
        omegaBlist.append(float(omegaB))
        hList.append(float(h))
        nsList.append(float(ns))
        s8list.append(float(s8))

        param_table = Table([omegaMlist, omegaBlist, hList, nsList, s8list],
                            names=('omegaM', 'omegaB', 'h', 'ns', 's8')) # astropy table
        
    k_min = 1e-5
    k_max = 1e3
    for i in range(len(param_table)):
        if os.path.exists(f'linearCF/linearCF{i}.dat'):
            continue

        print(f'Computing linear CF for parameter set {i}')

        parameters = param_table[i]
        omegaM, omegaB, h, ns, s8 = parameters['omegaM'], parameters['omegaB'], parameters['h'], parameters['ns'], parameters['s8']
        cosmoLin = cosmology(omegaM, omegaB, h, ns, s8)

        linCF = corr(cosmoLin, k_min, k_max)

        linCF.write(f'linearCF/linearCF{i}.dat', format='ascii', overwrite=True)  
