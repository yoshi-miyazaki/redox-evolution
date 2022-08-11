from   func_system     import *
from   func_masses     import *
from   func_partition  import *
from   func_atmosphere import *
from   func_plot       import *
import sys

import matplotlib.pyplot       as plt
from   matplotlib              import rc
from   matplotlib.font_manager import FontProperties
from   matplotlib.ticker       import MultipleLocator, FormatStrFormatter

plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.linewidth']= 1.0
plt.rcParams['font.size']     = 14
plt.rcParams['figure.figsize']= 4*1.414*4, 4*1
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{sfmath}')


"""
This file models the redox evolution during planet growth.
The magma ocean covers the "metal pond" before the next impactor arrives.
"""

# GLOBAL VARIABLEs


def simulation(iron_seg = False, f_core = .12, melt_factor = 30., fe_ratio = .45, h_frac = 0):
    # ----- [ initial condition ] -----

    rpl = 1000e3  # initial embryo size (radius in m)

    fe_mantle, fe_core = chem_mass_fe(f_core, fe_ratio, 0.1866)
    ni_mantle, ni_core = chem_mass_ni(0.05, f_core, 0.01091)

    xinit_mantle = np.array(
        [0.0954e2, 0.0102e2, 0.1070e2, 60.6e-4, 2623e-4, fe_mantle, 0.000513e2, ni_mantle, 18.3e-7, 345e-7])

    xinit_core = np.array([fe_core, ni_core])

    # ----- [ code ] -----
    ''' mass '''
    rc = core_radius(rpl, f_core)
    d_mantle = rpl - rc  # mantle depth
    M_mantle = shell_mass(rhom, rpl, d_mantle)  # calculate mass of initial planetesimal's mantle
    M_core = shell_mass(rhoc, rc, rc)

    ''' composition '''
    # calc the molar composition of mantle per 1 kg
    mole_mantle_unit = set_initial_mantle_composition_from_element(xinit_mantle)
    n_mantle = M_mantle * mole_mantle_unit  # convert mass to molar amount

    # calc the molar composition of core per 1 kg
    mole_core_unit = set_initial_core_composition(xinit_core)
    n_core = M_core * mole_core_unit  # convert mass to molar amount
    print("Fe in metal: ", n_core[nFe] / (n_mantle[nFe] + n_core[nFe]))

    # physical
    l_rpl, l_dm, l_Peq, l_Teq, l_DSi, l_fO2 = np.array([]), np.array([]), np.array([]), np.array([]), np.array(
        []), np.array([])
    l_sil, l_met = n_mantle, n_core
    l_Mm, l_Mc = M_mantle, M_core

    # solve for growth:
    # -- growing the planetesimal to a planetary embryo
    count = 0
    while (1):
        count += 1
        l_rpl = np.append(l_rpl, rpl)
        # print(count)

        if (rpl >= 6000e3 or count > 1000):
            l_Teq = np.append(l_Teq, l_Teq[-1])
            l_Peq = np.append(l_Peq, l_Peq[-1])
            print(rpl)
            break

        ''' 
        planetesimal growth 

        assume the size of impactor (20% radius of embryo)
        '''
        # calculate delivered amounts of material
        r_impactor = rpl / 5
        c_impactor = core_radius(r_impactor, f_core)

        # the core of the delivered planetesimal
        M_core_delivered = shell_mass(rhoc, c_impactor, c_impactor)
        n_core_delivered = M_core_delivered * mole_core_unit

        # the mantle of the delivered planetesimal
        M_mantle_delivered = shell_mass(rhom, r_impactor, r_impactor - c_impactor)
        n_mantle_delivered = M_mantle_delivered * mole_mantle_unit

        # the total of the delivered material
        M_delivered = M_mantle_delivered + M_core_delivered
        n_delivered = n_mantle_delivered + n_core_delivered

        Mpl = M_mantle + M_core + M_delivered
        g_acc = calculate_g(Mpl)

        ''' magma ocean '''
        h = calculate_h(melt_factor * 4. / 3 * np.pi * (r_impactor) ** 3, rpl)

        h_frac = shell_mass(rhom, rpl, h) / shell_mass(rhom, rpl, rpl - rc) if h_frac == 0 else h_frac

        ''' equilibrium '''
        # calculate compounds already present in mantle up till melted depth assuming a homogeneous mantle
        n_MO = h_frac * n_mantle
        n_partition = n_MO + n_delivered
        M_MO = mole2mass(n_MO)

        # calculate current pressure and temperature
        P_eq = Peq(g_acc, h)
        T_eq = Teq(P_eq * 1e9)  # pressure needs to be in Pa

        n_sil, n_met = partition_MO_impactor(n_partition, T_eq, P_eq)

        # solve for the partitioning of minor elements
        n_sil, n_met = partition_minor(n_sil, n_met, T_eq, P_eq)

        # solve for the segregation between FeO
        if(iron_seg):

            n_sil_new, n_met_new = seg_fe_phase(n_met, n_sil, T_eq, P_eq)

            n_sil, n_met = n_sil_new, n_met_new

        D = convert_D(n_sil, n_met)
        l_DSi = np.append(l_DSi, D[nSi])

        ''' why? '''
        dn = n_delivered - n_met
        xFe = n_met[nFe] / np.sum(n_met[nMg:])
        xSi = n_met[nSi] / np.sum(n_met[nMg:])
        xFeO = n_sil[nFe] / np.sum(n_sil[nMg:])
        xSiO2 = n_sil[nSi] / np.sum(n_sil[nMg:])

        # print(count, "\t Fe ratio: ", n_met[nFe]/n_delivered[nFe], "\t Si: ", n_met[nSi]/n_delivered[nSi], " \t org: " , n_MO[nFe]/n_MO[nSi], "\t", calc_KdSi(T_eq, P_eq), "  calc:", xSi*xFeO*xFeO/xSiO2/xFe/xFe)

        ''' atmosphere interaction '''
        # calculate oxygen fugacity
        x_FeO = n_sil[nFe] / np.sum(n_sil[nMg:])
        x_Fe = n_met[nFe] / np.sum(n_met[nMg:])

        # convert fO2 to bars, assuming temperature of the system is the temperature at 0 GPa pressure.
        fO2_now = calculate_ln_o_iw_fugacity(x_FeO, x_Fe)
        fO2_bar = fO2_fromIW(np.power(10., fO2_now), Teq(0))

        l_fO2 = np.append(l_fO2, fO2_now)

        # if (0):

        ''' update mantle & core compositions '''
        # add moles in metal phase to total moles of each element in the melt pond (which eventually sinks down into the planetary core)
        n_core += n_met
        n_mantle = n_mantle * (1 - h_frac) + n_sil

        # recalculate core and mantle masses
        M_mantle = mole2mass(n_mantle)
        M_core = mole2mass(n_core)

        # increase planet size
        rc = mass2radius(M_core, rhoc)
        d_mantle = mass2shell_width(M_mantle, rhom, rc)
        rpl = d_mantle + rc

        ''' save results '''
        l_sil = np.vstack((l_sil, n_mantle))
        l_met = np.vstack((l_met, n_core))

        l_dm = np.append(l_dm, d_mantle)
        l_Mm = np.append(l_Mm, M_mantle)
        l_Mc = np.append(l_Mc, M_core)
        l_Peq = np.append(l_Peq, P_eq)
        l_Teq = np.append(l_Teq, T_eq)

    wt_mantle, wt_core, wtMgO, wtFeO, wtSiO2, rFe, rSi = convert_wt_percent(l_sil, l_met)
    l_KdFe = calc_KdSi(l_Teq, l_Peq)

    '''
    # major element chemistry
    ax[1].set(ylabel="Mantle composition [wt%]")
    ax[1].plot(l_rpl/1e3, wtMgO*100,          color="k")                  # MgO wt% in the mantle
    ax[1].plot(l_rpl/1e3, wtFeO*100,          color="k", linestyle=":")   # FeO
    ax[1].plot(l_rpl/1e3, wtSiO2*100,         color="k", linestyle="--")  # SiO2
    ax[1].plot(l_rpl/1e3, wt_core[:,nFe]*100, color="r", linestyle=":")   # Fe wt% in the core
    ax[1].plot(l_rpl/1e3, wt_core[:,nSi]*100, color="r", linestyle="--")  # Ni
    ax[1].plot(l_rpl/1e3, rFe*100,            color="b", linestyle=":")   #(Fe in mantle)/(Fe in entire planet)
    ax[1].plot(l_rpl/1e3, rSi*100,            color="b", linestyle="--")  #(Si ...)
    ax[1].set_ylim([0,60])

    '''

    sum_of_squares = ((wt_mantle[-1:, nV] - 113e-6) ** 2 / 113e-6) + \
                     ((wt_mantle[-1:, nNi] - 3000e-6) ** 2 / 3000e-6) + \
                     ((wt_mantle[-1:, nCo] - 200e-6) ** 2 / 200e-6) + \
                     ((wt_mantle[-1:, nCr] - 4500e-6) ** 2 / 4500e-6) + \
                     ((wt_mantle[-1:, nTa] - 36e-9) ** 2 / 36e-9) + \
                     ((wt_mantle[-1:, nNb] - 675e-9) ** 2 / 675e-9)

    print("Error Value: ", sum_of_squares * 1e4)

    print("Oxygen Fugacity: ", l_fO2[-1:], "\n")

    #return(sum_of_squares)
    return(sum_of_squares)


best_score = 1000
best_h_frac = 0
best_melt_factor = 0

for j in range(0,110):
    score = simulation(False, .12,j,.45)
    #print("Percent Err: ", (score2 - score)/score * 100)
    print("Melt Factor", j)
    print("score = ", score)
    if (score < best_score):
        best_score = score
        best_melt_factor = j

print("Best Score", best_score)
print("Best Melt Factor", best_melt_factor)
#print("Best h_frac", best_h_frac)



