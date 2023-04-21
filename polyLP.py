import numpy as np
from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial

from readTables import txt_to_table


def CFpolyFit(r, xi, rmin, rmax, deg):
    rMask = (r <= rmax) & (r >= rmin)
    poly = Polynomial.fit(r[rMask], xi[rMask], deg)
    
    return poly

def polyLP(r, xi, rmin, rmax, deg):
    poly = CFpolyFit(r, xi, rmin, rmax, deg)
    dPoly = poly.deriv()
    ddPoly = dPoly.deriv()
    criticalPoints = dPoly.roots()
    realCP = criticalPoints[np.isreal(criticalPoints)]
    dipPeak = np.real(realCP[(realCP >= rmin) & (realCP <= rmax)])
    if len(dipPeak) >= 2:
        if ddPoly(dipPeak[0]) >= 0:
            lp = (dipPeak[0] + dipPeak[1])/2.
            return lp
        elif len(dipPeak) > 2:
            lp = (dipPeak[1] + dipPeak[2])/2.
            return lp
        else:
            return 0
    else:
        return 0
    
def plot3poly(number):
    linCFfile = ascii.read(f'linearCF/linearCF{number}.dat')  
    linCF = Table(linCFfile)
    
    nonLin = txt_to_table(number, 0) # z = 0
    nonLin3 = txt_to_table(number, 3) # z = 3
    
    dip, peak, lp, ip = scales(linCF)
    
    # Plotting
    fig = plt.figure(figsize=[5, 2.5], dpi=500)

    spl = UnivariateSpline(linCF['r'], linCF['xi'], k=4, s=0) # degree 4 spline fit

    ## Add vertical/horizontal width in plot ##
    r_min, r_max = dip - 50, peak + 50
    xi_max = spl(peak)*1.2
    if np.min(linCF['xi']) <= 0:
        xi_min = np.min(linCF['xi'])*1.4
    else:
        xi_min = np.min(linCF['xi'])*0.8
        
    z3poly = CFpolyFit(nonLin3['r'], nonLin3['xi'], 60, 140, 9)
    z0poly = CFpolyFit(nonLin['r'], nonLin['xi'], 60, 140, 9)

    rr = np.linspace(r_min, r_max, 1000)
    plt.plot(rr, z3poly(rr), lw = 1, ls = '--', zorder=1000, label='Quijote z=3 Polyfit', c='darkred')
    plt.plot(rr, z0poly(rr), lw = 1, ls='--', zorder=1000, label='Quijote z=0 Polyfit', c='orange')

    plt.plot(linCF['r'], linCF['xi'], c='r', label='Linear z=0 CF')

    plt.vlines([dip, peak], ymin=xi_min, ymax=xi_max, color='r', lw=0.5, ls='-.', label='Dip/peak')
    plt.vlines(lp, ymin=xi_min, ymax=xi_max, color='b', lw=0.5, ls='-.', label='LP')
    plt.vlines(ip, ymin=xi_min, ymax=xi_max, color='g', lw=0.5, ls='-.', label='IP')
    plt.scatter([dip, peak, lp, ip], [spl(dip), spl(peak), spl(lp), spl(ip)], c=['r', 'r', 'b', 'g'], s=2, zorder=1000)

    plt.plot(nonLin['r'], nonLin['xi'], label='Quijote z=0 CF', lw=2, c='b')
    plt.plot(nonLin3['r'], nonLin3['xi'], label='Quijote z=3 CF', lw = 2, c='lime')






    # figure parameters
    plt.xlim(r_min, r_max)
    plt.ylim(xi_min, xi_max)


    plt.xlabel('r  $[h^{-1}$ Mpc]', fontsize=6)
    plt.ylabel('$\\xi(r)$', fontsize=6)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.title('Correlation Function', fontsize=8)
    p = plt.legend(fontsize=4)