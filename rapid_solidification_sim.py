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
import multiprocessing as mp
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.linewidth']= 1.0
plt.rcParams['font.size']     = 14
plt.rcParams['figure.figsize']= 4*1.414*4, 4*2
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{sfmath}')

# Final core/mass ratio

"""
This file models the redox evolution during planet growth.
The magma ocean covers the "metal pond" before the next impactor arrives.

It models the rapid_solidification.py file into it's own function to 
change multiple variables and simulations
"""

# GLOBAL VARIABLES


def simulation(ni_mantle_per = 0.05, iron_seg = False, f_core = .12, melt_factor = 30., fe_ratio = .55, h_frac = 0.):
    # ----- [ initial condition ] -----

    rpl = 1000e3  # initial embryo size (radius in m)

    ''' composition '''
    # calc Fe wt% in the core/mantle based on the init conditions by Rubie et al. (2015)
    fe_mantle, fe_core = chem_mass_fe(f_core, fe_ratio, 0.1866)
    ni_mantle, ni_core = chem_mass_ni(ni_mantle_per, f_core, 0.01091)

    xinit_mantle = np.array(
        [0.0954e2, 0.0102e2, 0.1070e2, 60.6e-4, 2623e-4, fe_mantle, 0.000513e2, ni_mantle, 345e-7, 18.3e-7])

    xinit_core = np.array([fe_core, ni_core])

    # ----- [ code ] -----
    ''' mass '''
    rc       = core_radius(rpl, f_core)
    d_mantle = rpl - rc  # mantle depth
    M_mantle = shell_mass(rhom, rpl, d_mantle)  # calculate mass of initial planetesimal's mantle
    M_core   = shell_mass(rhoc, rc, rc)

    ''' composition '''
    # calc the molar composition of mantle per 1 kg
    mole_mantle_unit = set_initial_mantle_composition_from_element(xinit_mantle)
    n_mantle         = M_mantle * mole_mantle_unit  # convert mass to molar amount

    # calc the molar composition of core per 1 kg
    mole_core_unit   = set_initial_core_composition(xinit_core)
    n_core           = M_core * mole_core_unit  # convert mass to molar amount
    #print("Fe in metal: ", n_core[nFe] / (n_mantle[nFe] + n_core[nFe]))

    # save results
    l_rpl, l_dm, l_Peq, l_Teq, l_DSi, l_fO2 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    l_sil, l_met = n_mantle, n_core
    l_Mm, l_Mc = M_mantle, M_core

    ''' solve for growth '''
    # -- growing the planetesimal to a planetary embryo
    count = 0
    while (1):
        count += 1
        l_rpl = np.append(l_rpl, rpl)

        if (rpl >= 6000e3 or count > 1000):
            l_Teq = np.append(l_Teq, l_Teq[-1])
            l_Peq = np.append(l_Peq, l_Peq[-1])
            #print(rpl)
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
            for i in range(1):
                n_sil, n_met = seg_fe_phase(n_met, n_sil, T_eq, P_eq)

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
                     ((wt_mantle[-1:, nNb] - 675e-9) ** 2 / 675e-9) + \
                     ((wtFeO[-1] - 0.18) ** 2 / 0.18) + \
                     ((wtSiO2[-1] - 0.43) ** 2 / 0.43)

    #print("Error Value: ", sum_of_squares * 1e4)

    #print("Oxygen Fugacity: ", l_fO2[-1:], "\n")

    #return(sum_of_squares)

    return sum_of_squares * 1e4, l_fO2[-1:]

def rgb_to_hex(rgb):
    return '#' + ( '%02x%02x%02x' % rgb )

'''
This Test is tailored towards two components.
1. The nickel percent in the core
2. The melt factor
It aims to minimize the error as it simulates 
between the different combined amounts
'''
def test1(connection):
    ni_mf_scores = np.zeros((11, 100))

    for i in range(1, 10):
        ni_per = i / 100.
        print("Test 1 running", (i-1)/10. * 100, "% | Ni_percent =", ni_per)
        for j in range(10, 110):

            score, _ = simulation(ni_per, False, .12, j, .45)
            ni_mf_scores[i, j - 10] = score

    connection.send(ni_mf_scores)
    return ni_mf_scores

'''
Test2 is focused on h_frac and it's effects on the score relative to 
a changing melt factor. We want to see if h_frac has any strong effect on the score.
'''
def test2(connection):
    hf_mf_scores = np.zeros((21, 100))
    for i in range(1,20):
        h_frac = i * 5. / 100.
        print("Test 2 running", round((i-1) / 19. * 100, 1), "% | h_frac = ", h_frac)
        for j in range(10, 110):
            score, _ = simulation(0.05, False, 0.12, j, .55, h_frac)
            hf_mf_scores[i, j -10] = score
    connection.send(hf_mf_scores)
    return hf_mf_scores

'''
Test 3 uses changing values for the impactor core.
'''
def test3(connection):
    fe_per_mf_scores = np.zeros((23, 100))
    for i in range(12, 22):
        fe_core = i/100.
        print("Test 3 running", round((i - 12) /10. *100, 1), "% | fe_core = ", fe_core)
        for j in range(10, 110):
            score, _ = simulation(0.05, False, fe_core, j)
            fe_per_mf_scores[i, j-10] = score
    connection.send(fe_per_mf_scores)
    return fe_per_mf_scores

'''
Test 4 focuses on the difference between the scores with
or without using the iron_seg code.
'''
def test4(connection):
    fe_per_score= np.zeros((23, 50, 6))
    for i in range(12, 22):
        fe_core = i / 100.
        print("Test 4 running", round((i - 12) / 10. * 100, 1), "% | fe_core = ", fe_core)
        for j in range(0, 50):
            fe_raw, ox_raw = simulation(0.05, False, fe_core, j)
            fe_seg, ox_seg = simulation(0.05, True, fe_core, j)
            fe_per_score[i, j] = [fe_raw, fe_seg, fe_seg - fe_raw,
                                  ox_raw, ox_seg, ox_seg - ox_raw]
    connection.send(fe_per_score)
    return fe_per_score

'''
Changes Fe_ration
'''
def test5(connection):
    fe_rat_score = np.zeros((66, 100, 2))
    for i in  range(1, 13):
        fe_ratio = i/100. * 5.
        print("Test 5 running", round((i-1)/13. * 100, 1), "% | fe_ratio = ", fe_ratio)
        for j in range(10, 110):
            fe, ox = simulation(0.05, False, .12, j, fe_ratio)
            fe_rat_score[i, j-10] = [fe, ox]

    connection.send(fe_rat_score)
    return fe_rat_score




fig, ax = plt.subplots(2,4)
ax[0,0].set(ylabel="Sim Score")
ax[0,1].set(ylabel="Sim Score")
ax[0,2].set(ylabel="Sim Score")
ax[0,3].set(ylabel="Sim Score")

ax[1,0].set(ylabel="Diff. Scores")
ax[1,1].set(ylabel="Sim Score")
ax[1,2].set(ylabel="Diff. Scores")
ax[1,3].set(ylabel="Oxygen Fugacity")

ax[0,0].set(xlabel="Altering Nickel in Core")
ax[0,1].set(xlabel="Altering h_frac")
ax[0,2].set(xlabel="Altering size of core")
ax[0,3].set(xlabel="Varying fe_ratio")

ax[1,0].set(xlabel="Difference in Score (Segreg - Regular)")
ax[1,0].set(xlabel="With and without Segregation (Score)")
ax[1,0].set(xlabel="Difference in O_fug with Alterated - Regular")
ax[1,0].set(xlabel="With and without Segregation (Oxygen Fug)")

plt.tight_layout()

if __name__ == "__main__":

    testing_order = test5, test4, test1, test3, test2
    recv_data = []
    test_count = 0

    for test in testing_order:
        recv, comm = mp.Pipe()
        recv_data.append(recv)
        run = mp.Process(target= test, args=(comm,))
        run.start()

    data5 = recv_data[test_count].recv()
    print("> > > Received Test 5 results")
    col = 220
    for i in range(1, 13):
        color = (col, col, col)
        ax[0,3].plot(range(10, 110), data5[i, :, 0], color = rgb_to_hex(color), linestyle="-")
        col -= 12
    #ax[0,3].set_ylim([100, 800])
    plt.savefig("./sim.pdf")
    test_count += 1

    data4 = recv_data[test_count].recv()
    print("> > > Received Test 4 results")
    col = 220
    for i in range(12, 22):
        color = (col, col, col)
        color1 = (0, col, 0)
        color2 = (col, 0, 0)
        ax[1,0].plot(range(0, 50), data4[i, :, 2], color=rgb_to_hex(color), linestyle="-" )
        ax[1,1].plot(range(0, 50), data4[i, :, 0], color=rgb_to_hex(color1), linestyle="-")
        ax[1,1].plot(range(0, 50), data4[i, :, 1], color=rgb_to_hex(color2), linestyle="-")
        ax[1,2].plot(range(0, 50), data4[i, :, 5], color=rgb_to_hex(color), linestyle="-")
        ax[1,3].plot(range(0, 50), data4[i, :, 3], color=rgb_to_hex(color1), linestyle="-")
        ax[1,3].plot(range(0, 50), data4[i, :, 4], color=rgb_to_hex(color2), linestyle="-")
        col -= 16
    ax[1,1].set_ylim([100,800])
    plt.savefig("./sim.pdf")
    test_count += 1

    data1 = recv_data[test_count].recv()
    print("> > > Received Test 1 Results")
    color = (220, 220, 220)
    for i in range(1, 10):
        (a, b, c) = color
        color = (a - 20, b - 20, c - 20)
        ax[0,0].plot(range(10, 110), data1[i, :], color=rgb_to_hex(color), linestyle="-")
    ax[0,0].set_ylim([300, 1300])
    plt.savefig("./sim.pdf")
    test_count += 1

    data3 = recv_data[test_count].recv()
    print("> > > Received Test 3 Results")
    color = (220,220,220)
    for i in range(12, 22):
        (a, b, c) = color
        color = (a - 16, b - 16, c - 16)
        ax[0,2].plot(range(10,110), data3[i,:], color=rgb_to_hex(color), linestyle="-")
    ax[0,2].set_ylim([100, 1300])
    plt.savefig("./sim.pdf")
    test_count += 1

    data2 = recv_data[test_count].recv()
    print("> > > Received Test 2 Results")
    color = (220, 220, 220)
    for i in range(1,20):
        (a,b,c) = color
        color = (a - 5, b - 5, c- 5)
        ax[0,1].plot(range(10, 110), data2[i, :], color=rgb_to_hex(color), linestyle="-")
    ax[0,1].set_ylim([100, 1300])
    plt.savefig("./sim.pdf")
    test_count += 1

