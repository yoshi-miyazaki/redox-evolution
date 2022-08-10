import sys
from numba import njit
from func_chemistry import *


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
    elif (metal == "Cr"):
        a, b, c = -0.17, -2730, 0.
    elif (metal == "Co"):
        a, b, c = 0.13, 2057, -57
    elif (metal == "Ta"):
        a, b, c = 0.84, -13806, -115
    elif (metal == "Nb"):
        a, b, c = 2.66, -14032, -199
    elif (metal == "FeO"):
        return 70
    else:
        print("calc_Kd: no partition information available for", metal)

    logKd = a + b / T + c * P / T

    return np.power(10., logKd)


@njit
def calc_KdSi(T, P):
    # P^2 & P^3 terms are necessary to describe the partition of Si
    # Kd = np.exp(2.98 - 15934/T + (-155 * P + 2.26 * P**2 - 0.011 * P**3) / T)
    logKd = (2.98 - 15934 / T + (-155 * P + 2.26 * P ** 2 - 0.011 * P ** 3) / T)

    return np.power(10., logKd)


# @njit
def partition_MO_impactor(molep, T_eq, P_eq):
    '''
    Partition major elements between silicate and metal phases.
    The equilibrium reactions occurring are:
    
    2 Fe + SiO2 <--> 2 FeO + Si
      Fe + NiO  <-->   NiO + Fe

    For simplicity, we ignore the contribution of minor elements in mass balance.
    
    input --
    molep: mole amount of impactor+magma ocean 
    P_eq:  equilibrium pressure = base of magma ocean = metal pond
    T_eq:  equilibrium temperature 
    '''

    # calc partition coefficients
    Kd_Ni = calc_Kd("Ni", T_eq, P_eq)  # Ni
    Kd_Si = calc_KdSi(T_eq, P_eq)
    Kd_Cr = calc_Kd("Cr", T_eq, P_eq)

    # determine the range
    tot_Si, tot_Fe = molep[nSi], molep[nFe]

    # set the range for bisection search
    # x0, x1 represents the mole of Fe in the metal phase
    # considering the reaction, 
    x0 = tot_Fe * 0.001
    x1 = (tot_Fe - (molep[nO] - molep[nMg] - molep[nAl] * 1.5 - molep[nSi] * 2 - molep[nNi] - molep[nCo] - molep[nCr]))*1.
    xL = x1*1.
    
    f0 = df(x0, molep, Kd_Ni, Kd_Si, Kd_Cr)
    f1 = df(x1, molep, Kd_Ni, Kd_Si, Kd_Cr)

    if (f0 * f1 > 0):
        xA = (x0 + x1) * .5
        fA = df(xA, molep, Kd_Ni, Kd_Si, Kd_Cr)
        print("partition.py: bisection range error *: ", x0, x1, "\t", f0, f1, "\t", fA)

    eps = 1e-8
    while (np.abs(f1 - f0) > eps):
        xA = (x0 + x1) * .5
        fA = df(xA, molep, Kd_Ni, Kd_Si, Kd_Cr)

        if (f0 * fA < 0):
            x1, f1 = xA, fA
        else:
            x0, f0 = xA, fA

    # mole_sil, mole_met = massbalance_metFe(xA, molep, Kd_Ni, Kd_Si)
    mole_sil, mole_met = massbalance_metFe(xA, molep, Kd_Ni, Kd_Si, Kd_Cr)

    # restore minor elements
    for i in range(Nmol):
        if (i not in list_major):
            mole_sil[i] = molep[i]

    # print("r ", x1/xL, x1, xL)

    return mole_sil, mole_met


# @njit
def df(met_Fe, molep, Kd_Ni, Kd_Si, Kd_Cr):
    '''
    Returns the difference between the actual Kd(Si-Fe) and calculated value for Kd(Si-Fe). 

    input --
    met_Fe: assumed amount of Fe in the metal phase
    molep:  elemental abundance of metal + silicate phases
    Kd_Ni:  equilibrium constant for the Fe+NiO = Ni+FeO reaction
    '''

    tot_Mg, tot_Al, tot_Si, tot_Fe, tot_Ni,  tot_Cr = molep[nMg],molep[nAl],molep[nSi],molep[nFe],molep[nNi],molep[nCr]
    tot_O = molep[nO]

    # Calculate Fe amount
    sil_Fe = tot_Fe - met_Fe

    # calc Ni amount
    sil_Ni = tot_Ni * sil_Fe / (sil_Fe + Kd_Ni * met_Fe)
    met_Ni = tot_Ni - sil_Ni

    # calc Cr amount
    sil_Cr = tot_Cr * sil_Fe / (sil_Fe + Kd_Cr * met_Fe)
    met_Cr = tot_Cr - sil_Cr

    # calc Si amount
    sil_Si = (tot_O - sil_Ni - sil_Fe - tot_Mg - tot_Al * 1.5 - sil_Cr) / 2.
    met_Si = tot_Si - sil_Si

    # assume all Mg is exsits as oxides
    sil_Mg = tot_Mg
    sil_Al = tot_Al

    # compare with Kd_Si
    tot_met = met_Si + met_Fe + met_Ni + met_Cr
    tot_sil = sil_Si + sil_Fe + sil_Ni + sil_Mg + sil_Al + sil_Cr

    xSi_met = met_Si / tot_met
    xSi_sil = sil_Si / tot_sil
    xFe_met = met_Fe / tot_met
    xFe_sil = sil_Fe / tot_sil

    return xSi_met * (xFe_sil * xFe_sil) / (xFe_met * xFe_met * xSi_sil) - Kd_Si


def massbalance_metFe(met_Fe, molep, Kd_Ni, Kd_Si, Kd_Cr):
    '''
    The amoount of Fe in the metal phase is calculated in `partition_MO_impactor`.
    Based of its result, the rest of the composition is determined based on mass balance

    input  -- met_Fe (Fe in metal phase) + total molar amount of O, Mg, Si, Fe, Ni
    output -- composition vector of silicate and metal phases
    '''

    tot_O, tot_Mg, tot_Al = molep[nO], molep[nMg], molep[nAl]
    tot_Si, tot_Fe, tot_Ni = molep[nSi], molep[nFe], molep[nNi]

    tot_Cr = molep[nCr]

    # calc Fe, Ni amount 
    sil_Fe = tot_Fe - met_Fe
    sil_Ni = tot_Ni * sil_Fe / (sil_Fe + Kd_Ni * met_Fe)
    met_Ni = tot_Ni - sil_Ni

    # calc Cr amount
    sil_Cr = tot_Cr * sil_Fe / (sil_Fe + Kd_Cr * met_Fe)
    met_Cr = tot_Cr - sil_Cr

    # calc Si amount
    sil_Si = (tot_O - sil_Ni - sil_Fe - tot_Mg - tot_Al * 1.5 - sil_Cr) / 2.
    met_Si = tot_Si - sil_Si

    # store results
    mole_sil = np.zeros(Nmol)
    mole_met = np.zeros(Nmol)

    mole_sil[nSi], mole_sil[nFe], mole_sil[nNi], mole_sil[nCr] = sil_Si, sil_Fe, sil_Ni, sil_Cr
    mole_met[nSi], mole_met[nFe], mole_met[nNi], mole_met[nCr] = met_Si, met_Fe, met_Ni, met_Cr

    # assume all Mg is exsits as oxides
    mole_sil[nMg] = molep[nMg]
    mole_sil[nAl] = molep[nAl]
    mole_sil[nO] = molep[nO]

    return mole_sil, mole_met


# @njit
def partition_minor(n_sil, n_met, T_eq, P_eq):
    '''
    Calculate partitioing of minor elements between silicate and metal phases

    n_sil: major element compositions of the silicate phase
    n_met:                               the metal phase
    
    The total mole of minor elements in n_sil + n_met are correct, 
    but the partition between the two phases is not considered in partition_MO_impactor
    '''

    Kd_V  = calc_Kd("V" , T_eq, P_eq)
    Kd_Co = calc_Kd("Co", T_eq, P_eq)
    Kd_Ta = calc_Kd("Ta", T_eq, P_eq)
    Kd_Nb = calc_Kd("Nb", T_eq, P_eq)

    
    # calc total amount of major elements in silicate/metal phases
    #
    # when calculating [V] or [V2O3], the total mole of silicates or metal is needed.
    # but because minor elements contribute trivial amount compared to major elements,
    # we approximate tot_sil by mole of Mg, Al, Si, Fe, and Ni
    tot_sil, tot_met = 0., 0.
    for i in list_major:
        tot_sil += n_sil[i]
        tot_met += n_met[i]

    # find the amount of V in metal phase (x0, x1) by bisection search
    sil_Fe, met_Fe = n_sil[nFe], n_met[nFe]

    for i in list_minor:
        tot_el = n_sil[i] + n_met[i]

        if (i==nV):
            Kd_el, df_el, nO = Kd_V,  df_V,  1.5
        elif (i==nCo):
            Kd_el, df_el, nO = Kd_Co, df_Co, 1.
        elif (i==nTa):
            Kd_el, df_el, nO = Kd_Ta, df_Ta, 2.5
        elif (i==nNb):
            Kd_el, df_el, nO = Kd_Nb, df_Nb, 2.5
            
        found_met_el = bisection_search_minor(tot_el, met_Fe, sil_Fe, tot_met, tot_sil, Kd_el, df_el)
        n_met[i] = found_met_el
        n_sil[i] = tot_el - found_met_el

        n_met[nFe] -= found_met_el * nO
        n_sil[nFe] += found_met_el * nO

    return n_sil, n_met

def bisection_search_minor(tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El, df_El):
    # determine the search range
    x0 = tot_El * 0.001
    f0 = df_El(x0, tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El)

    x1 = el_met_max(x0, tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El, df_El)
    f1 = df_El(x1, tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El)

    if (f0 * f1 > 0):
        print("partition_minor: bisection range error: ", x0, x1, "\t", f0, f1)
        sys.exit()

    eps = 1e-6
    while (np.abs(f1 - f0) > eps):
        xA = (x0 + x1) * .5
        fA = df_El(xA, tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El)

        if (f0 * fA < 0):
            x1, f1 = xA, fA
        else:
            x0, f0 = xA, fA

    return xA


def el_met_max(x0, tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El, df_El):
    # find the larger end of the bisection search range
    f0 = df_El(x0, tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El)

    x1, f1 = tot_El, f0
    f1 = df_El(x1*0.999, tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El)
    
    while (f0 * f1 > 0):
        x1 *= 0.9
        f1 = df_El(x1, tot_El, met_Fe, sil_Fe, tot_met, tot_sil, Kd_El)

    #print(x1, f1, "\t", f0)

    return x1


def df_V (met_V, tot_V, met_Fe, sil_Fe, tot_met, tot_sil, Kd_V):
    '''
    Compare the difference between  Kd(V-Fe) and the calculated value for K_d(V-Fe).  
    (Root of function will be found at the correct value of K_d)

    input: assumed molar amount of V in metal phase, 
    -      Fe in metal, Fe in silicate, total cataion in metal, and total cation in silicatet phase
    

    stoichiomery:
    V - V2O3 + 3Fe = 3FeO + 2V

    '''
    # V as V2O3
    sil_V = tot_V - met_V

    xV_met = met_V / (tot_met + met_V)
    xV_sil = sil_V / (tot_sil + sil_V)
    xFe_met = met_Fe / (tot_met + met_V)
    xFe_sil = sil_Fe / (tot_sil + sil_V)

    return xV_met * xFe_sil ** 1.5 / (xV_sil * xFe_met ** 1.5) - Kd_V

def df_Co(met_Co, tot_Co, met_Fe, sil_Fe, tot_met, tot_sil, Kd_Co):
    '''
    stoichiomery:
    V - V2O3 + Fe = FeO
    Co - CoO + Fe = FeO + Co
    '''
    # V as V2O3
    sil_Co = tot_Co - met_Co

    xCo_met = met_Co / (tot_met + met_Co)
    xCo_sil = sil_Co / (tot_sil + sil_Co)
    xFe_met = met_Fe / (tot_met + met_Co)
    xFe_sil = sil_Fe / (tot_sil + sil_Co)

    return xCo_met * xFe_sil / (xCo_sil * xFe_met) - Kd_Co

def df_Ta(met_Ta, tot_Ta, met_Fe, sil_Fe, tot_met, tot_sil, Kd_Ta):
    '''
    stoichiomery:
    V - V2O3 + Fe = FeO
    Co - CoO + Fe = FeO + Co
    Ta - TaO2 + 2Fe = 2FeO + Ta
    '''
    # V as V2O3
    sil_Ta = tot_Ta - met_Ta

    xTa_met = met_Ta / (tot_met + met_Ta)
    xTa_sil = sil_Ta / (tot_sil + sil_Ta)
    xFe_met = met_Fe / (tot_met + met_Ta)
    xFe_sil = sil_Fe / (tot_sil + sil_Ta)

    return xTa_met * xFe_sil ** 2 / (xTa_sil * xFe_met ** 2) - Kd_Ta

def df_Nb(met_Nb, tot_Nb, met_Fe, sil_Fe, tot_met, tot_sil, Kd_Nb):
    '''
    stoichiomery:
    V - V2O3 + Fe = FeO
    Co - CoO + Fe = FeO + Co
    Nb (assuming NbO) = NbO + Fe = FeO + Nb
    '''
    # V as V2O3
    sil_Nb = tot_Nb - met_Nb

    xNb_met = met_Nb / (tot_met + met_Nb)
    xNb_sil = sil_Nb / (tot_sil + sil_Nb)
    xFe_met = met_Fe / (tot_met + met_Nb)
    xFe_sil = sil_Fe / (tot_sil + sil_Nb)

    return xNb_met * xFe_sil / (xNb_sil * xFe_met) - Kd_Nb

def df_FeO(met_Fe, tot_FeO, tot_FeO_onehalf,  tot_sil, Kd_onehalf):
    xFe = 1
    tot_sil -= met_Fe
    xFeO = (tot_FeO - 3 * met_Fe) / tot_sil
    xFeO_onehalf = (tot_FeO_onehalf + 2 * met_Fe) / tot_sil

    return xFeO ** 3 / xFeO_onehalf ** 2 / xFe - Kd_onehalf

def seg_fe_phase(n_met, n_sil, T_eq, P_eq):
    '''
    Fe + 2FeO1.5 = 3FeO

    [FeO]**3 / [Fe] / [Fe2O3]**2

    1.0 o   1.5 fe    <- Fe2O3
    1.0 o   1.0 fe    <- FeO
            1.2 fe

    Fe + 2FeO1.5 -> 3FeO
    '''

    tot_sil, tot_met = 0., 0.
    for i in list_major:
        tot_sil += n_sil[i]
        tot_met += n_met[i]

    Kd_FeO = calc_Kd("FeO", T_eq, P_eq)

    met_Fe = n_met[nFe]
    sil_Fe = n_sil[nFe]
    available_O = n_sil[nO] - n_sil[nMg] - n_sil[nNi] - n_sil[nAl] * 1.5 - n_sil[nSi] * 2.0 - n_sil[nCr] - n_sil[nCo]

    # (1.5x + 1.0(total_fe - x) = total
    FeO_onehalf = 2. * (available_O - sil_Fe)
    FeO = sil_Fe - FeO_onehalf

    x0 = 1e-8
    f0 = df_FeO(x0, FeO, FeO_onehalf, tot_sil, Kd_FeO)

    x1 = sil_Fe #+ 1e23
    f1 = df_FeO(x1, FeO, FeO_onehalf, tot_sil, Kd_FeO)

    if (f0 * f1 > 0):
        print("Partition Iron Segregation Bisection Error, same sign ", f0, f1)
        print(n_sil)

        sil_Fe = n_sil[nFe]
        sil_Mg = n_sil[nMg]
        sil_Si = n_sil[nSi]
        sil_Ni = n_sil[nNi]
        sil_Al = n_sil[nAl]
        sil_Cr = n_sil[nCr]
        sil_Co = n_sil[nCo]
        sil_O  = n_sil[nO]

        print(sil_O - sil_Co - sil_Cr - sil_Si*2 - sil_Ni - sil_Al*1.5  - sil_Mg)
        print(sil_Fe)
        
        exit()

    eps = 1e-6

    while(np.abs(f1 - f0) > eps):
        xA = (x0 + x1) * .5
        #print(xA)
        fA = df_FeO(xA, FeO, FeO_onehalf, tot_sil, Kd_FeO)

        if (f0 * fA < 0):
            x1, f1 = xA, fA
        else:
            x0, f0 = xA, fA

    n_met[nFe] += xA
    n_sil[nFe] -= xA

    return n_sil, n_met



