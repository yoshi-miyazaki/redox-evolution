import numpy as np
import sys
from   numba          import njit
from   func_chemistry import *

# partition coefficients between metal and silicate phases
@njit
def calc_Kd(metal, T, P):
    '''
    Calculates Kd using constants given 
    in Rubie et al. (2015): Table S1 / Equation (3)
    '''
    if (metal == "Ni"):
        a, b, c = 1.06, 1553, -98
    elif (metal == "V"):
        a, b, c = -0.48, -5063, 0.
    else:
        print("calc_Kd: no partition information available for", metal)
    
    logKd = a + b/T + c*P/T
    
    return np.power(10., logKd)

@njit
def calc_KdSi(T, P):
    # P^2 & P^3 terms are necessary to describe the partition of Si
    #Kd = np.exp(2.98 - 15934/T + (-155 * P + 2.26 * P**2 - 0.011 * P**3) / T)
    logKd = (2.98 - 15934/T + (-155 * P + 2.26 * P**2 - 0.011 * P**3) / T)
    
    return np.power(10., logKd)

# @njit
def partition_MO_impactor(molep, T_eq, P_eq):
    '''
    Partition major elements between silicate and metal phases.
    The equilibrium reactions occurring are:
    
    2 Fe + SiO2 <--> 2 FeO + Si
    2 Ni + SiO2 <--> 2 NiO + Si

    For simplicity, we ignore the contribution of V in mass balance.
    '''

    # calc partition coefficients
    Kd_Ni = calc_Kd("Ni", T_eq, P_eq)  # Ni
    Kd_Si = calc_KdSi    (T_eq, P_eq)
    
    # determine the range
    tot_Si, tot_Fe = molep[nSi], molep[nFe]

    # set the range for bisection search
    # x0, x1 represents the mole of Fe in the metal phase
    # considering the reaction, 
    x0 = tot_Fe*0.001
    x1 = tot_Fe - (molep[nO] - molep[nMg] - molep[nAl]*1.5 - molep[nSi]*2 - molep[nNi])
    f0 = df(x0, molep, Kd_Ni, Kd_Si)
    f1 = df(x1, molep, Kd_Ni, Kd_Si)
    
    if (f0*f1 > 0):
        print("partition.py: bisection range error: ", x0, x1, "\t", f0, f1)
        sys.exit()
    
    eps = 1e-8
    while (np.abs(f1-f0) > eps):
        xA = (x0+x1)*.5
        fA = df(xA, molep, Kd_Ni, Kd_Si)

        if (f0*fA < 0):
            x1, f1 = xA, fA
        else:
            x0, f0 = xA, fA

    mole_sil, mole_met = massbalance_metFe(xA, molep, Kd_Ni, Kd_Si)
    
    # restore minor elements
    for i in range(Nmol):
        if (i not in list_major):
            mole_sil[i] = molep[i]

    #print(mole_sil+mole_met-molep)
    #sys.exit()
            
    return mole_sil, mole_met

#@njit
def df(met_Fe, molep, Kd_Ni, Kd_Si):
    '''
    Returns the difference between the actual Kd(Si-Fe) and calculated value for Kd(Si-Fe). 
    '''

    tot_O,  tot_Mg, tot_Al = molep[nO],  molep[nMg], molep[nAl]
    tot_Si, tot_Fe, tot_Ni = molep[nSi], molep[nFe], molep[nNi]
    
    # calc Ni amount 
    sil_Fe = tot_Fe - met_Fe
    sil_Ni = tot_Ni * sil_Fe / (sil_Fe + Kd_Ni * met_Fe)
    met_Ni = tot_Ni - sil_Ni

    # calc Si amount
    sil_Si = (tot_O - sil_Ni - sil_Fe - tot_Mg - tot_Al*1.5) /2.
    met_Si = tot_Si - sil_Si

    # assume all Mg is exsits as oxides
    sil_Mg = tot_Mg
    sil_Al = tot_Al
    
    # compare with Kd_Si
    tot_met = met_Si + met_Fe + met_Ni
    tot_sil = sil_Si + sil_Fe + sil_Ni + sil_Mg + sil_Al
    
    xSi_met = met_Si / tot_met
    xSi_sil = sil_Si / tot_sil
    xFe_met = met_Fe / tot_met
    xFe_sil = sil_Fe / tot_sil

    return xSi_met*(xFe_sil*xFe_sil) / (xFe_met*xFe_met * xSi_sil) - Kd_Si


def massbalance_metFe(met_Fe, molep, Kd_Ni, Kd_Si):
    '''
    The amoount of Fe in the metal phase is calculated in `partition_MO_impactor`.
    Based of its result, the rest of the composition is determined based on mass balance

    input:  met_Fe (Fe in metal phase) + total molar amount of O, Mg, Si, Fe, Ni
    output: composition vector of silicate and metal phases
    '''

    tot_O,  tot_Mg, tot_Al = molep[nO],  molep[nMg], molep[nAl]
    tot_Si, tot_Fe, tot_Ni = molep[nSi], molep[nFe], molep[nNi]
    
    # calc Fe, Ni amount 
    sil_Fe = tot_Fe - met_Fe
    sil_Ni = tot_Ni * sil_Fe / (sil_Fe + Kd_Ni * met_Fe)
    met_Ni = tot_Ni - sil_Ni

    # calc Si amount
    sil_Si = (tot_O - sil_Ni - sil_Fe - tot_Mg - tot_Al*1.5) /2.
    met_Si = tot_Si - sil_Si

    # store results
    mole_sil = np.zeros(Nmol)
    mole_met = np.zeros(Nmol)
    
    mole_sil[nSi], mole_sil[nFe], mole_sil[nNi] = sil_Si, sil_Fe, sil_Ni
    mole_met[nSi], mole_met[nFe], mole_met[nNi] = met_Si, met_Fe, met_Ni
    
    # assume all Mg is exsits as oxides
    mole_sil[nMg] = molep[nMg]
    mole_sil[nAl] = molep[nAl]
    mole_sil[nO]  = molep[nO]

    return mole_sil, mole_met



#@njit
def partition_minor(n_sil, n_met, T_eq, P_eq):
    '''
    Calculate partitioing of minor elements between silicate and metal phases
    '''
    
    Kd_V  = calc_Kd("V", T_eq, P_eq)

    # calc total amount of major elements in silicate/metal phases
    tot_sil, tot_met = 0., 0.
    for i in [nMg, nSi, nFe, nNi]:
        tot_sil += n_sil[i]
        tot_met += n_met[i]

    sil_Fe, met_Fe = n_sil[nFe], n_met[nFe]
    tot_V          = n_sil[nV] + n_met[nV]
    
    # determine the search range
    x0 = tot_V*0.0001
    f0 = dg(x0, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V)
    
    x1 = vmet_max(x0, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V)
    f1 = dg(x1, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V)

    if (f0*f1 > 0):
        print("partition_minor: bisection range error: ", x0, x1, "\t", f0, f1)
        sys.exit()
    
    eps = 1e-6
    while (np.abs(f1-f0) > eps):
        xA = (x0+x1)*.5
        fA = dg(xA, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V)

        if (f0*fA < 0):
            x1, f1 = xA, fA
        else:
            x0, f0 = xA, fA

    #print("search:", abs(x1-x0)/x0, "\t", f0, fA, f1)

    # restore results
    n_met[nV] = xA
    n_sil[nV] = tot_V - xA
    
    return n_sil, n_met

#@njit
def dg(met_V, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V):
    '''
    Compare the difference between  Kd(V-Fe) and the calculated value for K_d(V-Fe). 
    (Root of function will be found at the correct value of K_d)

    input: assumed molar amount of V in metal phase, 
    -      Fe in metal, Fe in silicate, total cataion in metal, and total cation in silicatet phase

    '''
    # V as V2O3
    sil_V   = tot_V - met_V
    
    xV_met  = met_V / (tot_met + met_V)
    xV_sil  = sil_V / (tot_sil + sil_V)
    xFe_met = met_Fe/ (tot_met + met_V)
    xFe_sil = sil_Fe/ (tot_sil + sil_V)
    
    return xV_met*xFe_sil**1.5 / (xV_sil*xFe_met**1.5) - Kd_V

def vmet_max(x0, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V):
    # find the larger end of the bisection search range
    f0 = dg(x0, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V)
    
    x1, f1 = tot_V, f0
    while (f0*f1 > 0):
        x1 *= 0.9
        f1 = dg(x1, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V)

    return x1
    
