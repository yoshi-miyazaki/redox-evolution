from numba import jit
''' 
dG (J/mol) as a function of T. Data taken from JANAF database and fit using a linear regression 
(by Yoshi Miyazaki)
'''
# Gibbs free energies of formation fit with a linear regression using data on JANAF database
@jit(nopython=True)
def GCO(T):
    dG = 0.05829 * T - 252.033
    return dG * 1e3


@jit(nopython=True)
def GCO2(T):
    dG = 7.43e-4 * T - 397.691
    return dG * 1e3

@jit(nopython=True)
def GH2O(Tin):
    dG = 0.0547*Tin - 246.56   # in kJ/mol. Tin is in [K].

    return dG*1e3  # in J/mol.


@jit(nopython=True)
def GCH4(T):
    dG = 0.1115725 * T - 92.3855
    return dG * 1e3


@jit(nopython=True)
def GFeO(T):
    dG = 0.05277 * T - 254.1475
    return dG * 1e3
