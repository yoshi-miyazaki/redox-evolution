'''
Define universal constans & material parameters 
'''

import numpy  as np
from   numba  import njit, jit, float32

# Universal constants
G = 6.674e-11
R = 8.31446  # universal gas constant


k = 3.926763924239811e-12 # using Mars' mass-g relationship

# material properties
rhom = 5000
rhoc = 8000
rho_mantle = 3000
rho_core = 8000



# molar masses (kg / mol)
molar_mass_h = 1e-3
molar_mass_c = 12e-3



# mathermatical conversion
tolog10 = np.log10(np.e)
