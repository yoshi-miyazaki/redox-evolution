from func_chemistry import *


def convert_wt_percent(l_mantle, l_core):

    # convert to mass from molar amount
    m_mantle = l_mantle * molar_mass
    m_core   = l_core   * molar_mass

    # total mass of each
    M_mantle = np.sum(m_mantle, axis=1)
    M_core = np.sum(m_core,   axis=1)
    
    # wt%
    wt_mantle = m_mantle/M_mantle[:,None]
    wt_core   = m_core  /M_core[:,None]

    # convert wt% to oxides
    # (mantle) FeO, SiO2, (core) Si
    wtMgO  = wt_mantle[:,nMg] * (molar_mass[nO]  +molar_mass[nMg])/molar_mass[nMg]
    wtFeO  = wt_mantle[:,nFe] * (molar_mass[nO]  +molar_mass[nFe])/molar_mass[nFe]
    wtSiO2 = wt_mantle[:,nSi] * (molar_mass[nO]*2+molar_mass[nSi])/molar_mass[nSi]
    
    # (e in mantle)/(e total)  for Fe and Si
    rFe = m_mantle[:,nFe]/(m_mantle[:,nFe]+m_core[:,nFe])
    rSi = m_mantle[:,nSi]/(m_mantle[:,nSi]+m_core[:,nSi])
    
    return wt_mantle, wt_core, wtMgO, wtFeO, wtSiO2, rFe, rSi


def convert_D(n_mantle, n_core):
    # total mole
    tot_mantle = np.sum(n_mantle * molar_mass)
    tot_core   = np.sum(n_core   * molar_mass)
    
    # molar %
    r_mantle = (n_mantle*molar_mass)/tot_mantle
    r_core   = (n_core  *molar_mass)/tot_core

    # distribution coeff.
    D = r_core/r_mantle

    return D
