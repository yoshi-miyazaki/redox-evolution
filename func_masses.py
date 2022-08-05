import math
import sys

from   func_system import *

''' mass '''
def body_mass(r_body, core_radius):
    return 4 / 3 * math.pi * (core_radius**3 * rho_core + (r_body - core_radius)**3 * rho_mantle)

# mass of shell with outer rapidus of `r_planet` and depth of `depth`.
# -- for sphere, set r_planet == depth
def shell_mass(rho, r_planet, depth):
    return rho * 4./3*math.pi * (r_planet**3 - (r_planet - depth)**3)

# takes percent of core, proportion of iron in core/mantle, and total percent
# returns fe of iron in core and mantle
def chem_mass_fe(r, p, t):
    fe_mantle = t * (1 - p) / (1 - r) * 100
    fe_core   = t * p / r * 100
    print("Iron Mantle, Iron Core: ", fe_mantle, fe_core)
    if (fe_mantle < 0 or fe_core < 0):
        print("Error: iron less than 0%")
        sys.exit()
    elif (fe_mantle > 100 or fe_core > 100):
        print("Error: iron more than 100%")
        sys.exit()
    return fe_mantle, fe_core

def chem_mass_ni(ni_core, r, t):
    ni_mantle = (t - ni_core * r) / (1 - r) * 100
    ni_core *= 100
    print("Nickel Mantle, Nickel Core: ", ni_mantle, ni_core)
    if (ni_mantle < 0 or ni_core < 0):
        print("Error: nickel less than 0%")
        sys.exit()
    elif (ni_mantle > 100 or ni_core > 100):
        print("Error: nickel more than 100%")
        sys.exit()
    return ni_mantle, ni_core

# calculate core radius, assuming the core mass is `frac_core` of the impactor
def core_radius(rpl, f_core):
    ''' needs fixing '''
    return (rpl**3 * f_core)**(1./3)

# calculate the sphere radius from mass and density
def mass2radius(M, rho):
    return np.power(M * 3./(4*np.pi*rho), 1./3)

# calculate the shell width from mass and density
def mass2shell_width(M, rho, r_in):
    return np.power(M * 3./(4*np.pi*rho) + r_in**3, 1./3) - r_in


''' equilibrium conditions (base of magma ocean) '''
def Peq(g, h):
    # returns P of the base of MO in GPa
    return rhom * g * h / 1e9


def Teq(Peq):
    # input: Peq (Pa)
    Tsol = 1661.2 * (1. + Peq/1.336e9)**(1 / 7.437)
    Tliq = 1982.1 * (1. + Peq/6.594e9)**(1 / 5.374)
    
    # return T of rheological transition
    f_rheo = 0.4
    return f_rheo*Tsol + (1-f_rheo)*Tliq


def calculate_g(M_p):
     # Calculate gravitational acceleration using g ~ M_p^(0.503) and M_Mars = 6.39e23 kg, g_Mars = 3.7 m / s 
    return M_p**0.503 * k


def calculate_h(melt_vol, r_planet):
    # Returns the depth of mantle that is melted by an impactor.
    return r_planet - (r_planet**3 - 3 / 4 / math.pi * melt_vol)**(1 / 3)


