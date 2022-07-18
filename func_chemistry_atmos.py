from func_system import *
from gibbse import *

''' volatile - basic information '''
Nvol = 4   # H2, H2O, CO, CO2 


''' oxygen fugacity '''
def calculate_ln_o_iw_fugacity(X_FeO, X_Fe):
    """Calculates fO2 - IW"""
    return 2 * np.log10(X_FeO / X_Fe)


def fO2_fromIW(fO2_dIW, T):
    ''' 
    convert fO2 in terms of [bar]. 
    The relation ln(fO2) = 2 ln(xFeO/xFe) shows the value in terms of [difference from IW]
    (by Yoshi Miyazaki)
    '''

    # log10(fO2) is -12 at 1200 degC, -8 at 1700 degC.
    # assume a log-linear relationship for now
    fO2_IW = 10**(-12 + (T-1473.15)/500*4.)
    
    # convert
    fO2    = fO2_IW * fO2_dIW
    return fO2


''' mass balance '''
def massbalance_CH(fO2, m_H, m_C):
    # assume the redox H & C would be reset by the presence of metal pond
    # input: fO2   (oxygen fugacity in bar)
    #        mol_H (total H
    rH2O_H2 = H2O_H2ratio(fO2, Teq(0))
    rCO2_CO = CO2_COratio(fO2, Teq(0))

    # ignore CH4 here because of its low concentration
    m_H2  = 0.5 * mol_H / (rH2O_H2 + 1) 
    m_H2O = 0.5 *(mol_H - 2*m_H2)
    m_CO  = m_C / (rCO2_CO + 1)
    m_CO2 = m_C - m_CO
    #m_CH4 = (mol_h - 2 * mol_H2 - 2 * mol_H2O) / 4

    n_atmos = np.array([m_H2, m_H2O, m_CO, m_CO2])
    
    return n_atmos

# oxidized : reduced compound ratios calculated using fugacity
def CO_Cratio(fO2, T):
    # C + 0.5 O2 <--> CO
    muCO = GCO(T)
    muC  = 0
    muO2 = 0
    
    dG  = muCO - muC - 0.5 * muO2
    Keq = np.exp(-dG / (R * T))
    
    rCO_xC = Keq * np.sqrt(fO2)
    return rCO_xC


def CO2_COratio(fO2, T):
    # CO + 0.5 O2 <--> CO2
    muCO2 = GCO2(T)
    muCO = GCO(T)
    muO2 = 0
    
    dG = muCO2 - muCO - 0.5 * muO2
    Keq = np.exp(-dG / (R * T))
    rCO2_CO = Keq * np.sqrt(fO2)
    
    return rCO2_CO


def CO2H2O_CH4ratio(fO2, T):
    # """Calculates equilibrium constant for the reaction CH4 + 2 O2 <--> CO2 + 2 H2O"""
    muCO2 = GCO2(T)
    muCH4 = GCH4(T)
    muH2O = GH2O(T)

    dG = muCO2 + 2 * muH2O - muCH4
    Keq = np.exp(-dG / (R * T))
    rspecies_XCH4 = Keq * fO2**2
    return rspecies_XCH4


def H2O_H2ratio(fO2, Tin):
    ''' solve for H2O/H2 ratio using equilibrium constant (by Yoshi Miyazaki)'''
    muH2O = GH2O(Tin)
    muO2  = 0.
    muH2  = 0.
    
    dG  = - muH2 - 0.5*muO2 + muH2O  # in J/mol
    Keq = np.exp(-dG/(R*Tin))       # convert to equilibrium constant
    rH2O_xH2 = Keq*np.sqrt(fO2)    # equilibrium relation
    
    return rH2O_xH2

# equilibrium constants for re-equilibrium
@jit(nopython=True)
def Keq_FeO_H2O(T):
    """Calculates equilibrium constant for the reaction FeO + H2 <--> Fe + H2O"""
    Keq = np.exp(-(GH2O(T) - GFeO(T)) / R / T)
    return Keq

@jit(nopython=True)
def Keq_FeO_CO2(T):
    """Calculates equilibrium constant for the reaction FeO + CO <--> Fe + CO2"""
    Keq = np.exp(-(GCO2(T) - GFeO(T) - GCO(T)) / R / T)
    return Keq

@jit(nopython=True)
def Keq_FeO_CH4(T):
    """Calculates equilibrium constant for the reaction FeO + CH4 <--> Fe + CO + 2 H2"""
    Keq = np.exp(-(GCO(T) - GFeO(T) - GCH4(T)) / R / T)
    return Keq
