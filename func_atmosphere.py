from func_chemistry_atmos import *


def atmos_reequilibrium():
    # volatile composition equilibrium with the metal pond
    n_atmos = massbalance_CH(fO2_bar, mol_h, mol_c)
        
    print(count, x_FeO, x_Fe, "\t", fO2_now)
    #sys.exit()
    
    # calculate moles of H in the magma ocean, assuming all H is present in the form of H2O.
    mol_h = 2 * h_frac * 0.01 * h_s * M_mantle  / (molar_mass_o + molar_mass_h * 2) + delivered_h
    
    # calculate moles of C in the magma ocean, assuming all C is present in the form of CO2.
    mol_c = h_frac * 0.01 * c_s * M_mantle  / (molar_mass_c + molar_mass_o * 2) + delivered_c
    
    
    mol_volatiles = mol_H2O + mol_H2 + mol_CO + mol_CO2 # + mol_CH4
    #print("fe, ni, si ", fe_sil, "\t", ni_sil, "\t", si_sil)
    #print("initial mol_H2, H2O ", mol_H2, mol_H2O, " CO2: ",mol_CO2, "\t r = ", rH2O_pond, " Fe", fe_sil)
    
    # calculate total amount of oxygen in atmosphere-MO system
    mol_fe_reeq = fe_sil
    mol_o_atmos = mol_fe_reeq + mol_H2O + mol_CO + mol_CO2 * 2
    #print("atmospheric oxygen", mol_o_atmos)
    
    # re-equilibrium between FeO and volatiles
    #print("for reeq Teq = ", T_eq, P_eq)
    xmetal = calc_metal(mol_fe_reeq, ni_sil, si_sil, (mol_mg+mol_al)*h_frac, 4000, 0) #T_eq, P_eq)
    mol_fe_mo = bisection_search("mol_fe_mo", mol_fe_reeq*.8, mol_fe_reeq*.99999, 1e-4, fe_sil, ni_sil, si_sil, mol_o_atmos, v_sil, (mol_mg+mol_al) * h_frac, mol_c, T_eq, mol_h)
    kd_CO2 = Keq_FeO_CO2(Teq(0))
    kd_H2O = Keq_FeO_H2O(Teq(0))
    #print("keq H2o", kd_H2O)
    #print("keq co2", kd_CO2)
    mol_fe_metal = mol_fe_reeq - mol_fe_mo

    #
    xmetal = .8
    conc_fe_mo = mol_fe_mo / (ni_sil + mol_fe_mo + si_sil + (mol_mg+mol_al) * h_frac + v_sil)
    H2O_to_H2  = kd_H2O * conc_fe_mo / xmetal
    mol_H2     = .5 * mol_h / (H2O_to_H2 + 1)   # ignore CH4 here because its concentration is low
    mol_H2O    = .5 * (mol_h - 2 * mol_H2)
    CO2_to_CO  = kd_CO2 * conc_fe_mo / xmetal
    mol_CO     = mol_c / (CO2_to_CO + 1)
    mol_CO2    = mol_c - mol_CO
    mol_volatiles = mol_H2O + mol_H2 + mol_CO + mol_CO2
    #print("percentage of iron removed", mol_fe_metal / mol_fe_reeq, "\t", mol_fe_metal, " mol of Fe created, ", mol_H2O, mol_H2)
    #print("r after = ", H2O_to_H2)

    #print("after mol CO2, mol CO, mol_CH4, mol_H2, mol_H2O, fe in metal, fe_sil", mol_CO2, mol_CO, mol_CH4, mol_H2, mol_H2O, mol_fe_metal, mol_fe_mo)

    #print("atmospheric composition", mol_CO, mol_CO2, mol_H2, mol_H2O)
    total_CO.append(mol_CO)
    total_CO2.append(mol_CO2)
    total_H2.append(mol_H2)
    total_H2O.append(mol_H2O)
    total_CH4.append(mol_CH4)

    # Fe transport from MO to core by re-equilibrium
    mols_fe_c += mol_fe_metal        # allow the Fe metal to go to planetary core and update size of planet
    mol_fe    -= mol_fe_metal        # remove metal phase Fe from mantle
    
    # once again calculate core mass, mantle mass, and planet radius
    new_core_mass  = convert_moles_to_mass(mols_fe_c, molar_mass_fe) + convert_moles_to_mass(mols_ni_c, molar_mass_ni) + convert_moles_to_mass(mols_si_c, molar_mass_si) + convert_moles_to_mass(mols_v_c, molar_mass_v)
    new_mantle_mass = convert_moles_to_mass(mol_fe, molar_mass_fe) + convert_moles_to_mass(mol_ni, molar_mass_ni) + convert_moles_to_mass(mol_si, molar_mass_si) + convert_moles_to_mass(mol_v, molar_mass_v) + convert_moles_to_mass(mol_o, molar_mass_o) + convert_moles_to_mass(mol_mg, molar_mass_mg) + convert_moles_to_mass(mol_al, molar_mass_al)

    # increase planet size
    rc = sphere_radius(new_core_mass, rho_core)
    new_mantle_depth = shell_width(new_mantle_mass, rhom, rc)
    rpl = new_mantle_depth + rc
    
    mol_volatiles_tot = mol_H2 + mol_H2O + mol_CO2 + mol_CO + mol_CH4
    fraction_H2O = mol_H2O / mol_volatiles_tot
    fraction_H2  = mol_H2  / mol_volatiles_tot
    final_fO2    = calculate_fugacity(fraction_H2O, fraction_H2, Keq_FeO_H2O(Teq(0)))
    final_fO2_list.append(final_fO2)
    #print("final fO2", final_fO2)

    r_H2O.append(mol_H2O/mol_H2) #(mol_H2O+mol_H2))


    return mol_H2
