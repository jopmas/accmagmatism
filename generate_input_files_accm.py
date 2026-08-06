import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

import xarray as xr

path = os.getcwd().split('/')
local_path = os.getcwd()
machine_path = '/'+path[1]+'/'+path[2]

#CLASSES
class MandyocLayer:
    def __init__(self, layer_label, rheology: type,
                 density, effective_viscosity_scale_factor=1.0, radiogenic_heat_production=0.0,
                 base_depth=0.0e3, Nx=()):
        """"
        This class creates a layer with the given properties
        layer_label: str
            Name of the layer
        rheology: class
            Rheological properties of the layer onbtained from LithologicalUnit class
        density: float
            Density of the layer [kg/m3]
        interface: NoneType or np.array
            Interface of the layer.
            If None, the interface will be setted after the creation of the Layer.
            If np.array, the interface is defined by the given array.
        effective_viscosity_scale_factor: float
            Scale factor for the effective vistocisty
        radiogenic_heat_production: float
            Radiogenic heat production of the layer [W/kg]
        base_depth: float
            Depth of the layer base [m]
        Nx: int
            Number of points in x direction
        """

        self.layer_label = layer_label
        self.rheology = LithologicalUnit(rheology)
        self.density = density
        self.effective_viscosity_scale_factor = effective_viscosity_scale_factor
        self.radiogenic_heat_production = radiogenic_heat_production
        self.rheology_name = self.rheology.name
        self.pre_exponential_constant = self.rheology.pre_exponential_constant
        self.power_law_exponent = self.rheology.power_law_exponent
        self.activation_energy = self.rheology.activation_energy
        self.activation_volume = self.rheology.activation_volume
        self.base_depth = base_depth
        self.Nx = Nx

        self.interface = np.ones(Nx)*base_depth

class LithologicalUnit:
    """"
    This class calls the respective rheological properties of the given mineral

    mineral_name: class
        Mineral rheology written in CamelCase. For example, WetOlivine, DryOlivine, WetQuartz
    """
    def __init__(self, mineral_name: type):
        self.mineral_name = mineral_name() # mineral_name is a class, so we need to call it to get the object
        self.name = self.mineral_name.name
        self.pre_exponential_constant = self.mineral_name.pre_exponential_constant
        self.power_law_exponent = self.mineral_name.power_law_exponent
        self.activation_energy = self.mineral_name.activation_energy
        self.activation_volume = self.mineral_name.activation_volume

class WetOlivine:
    """
    Wet olivine rheological properties
    """
    def __init__(self):
        self.name = 'wet_olivine'
        self.pre_exponential_constant = 1.393e-14
        self.power_law_exponent = 3
        self.activation_energy = 429.0e3
        self.activation_volume = 15.0e-6

class DryOlivine:
    """
    Dry olivine rheological properties
    """
    def __init__(self):
        self.name = 'dry_olivine'
        self.pre_exponential_constant = 2.4168e-15
        self.power_law_exponent = 3.5
        self.activation_energy = 540.0e3
        self.activation_volume = 25.0e-6

class WetQuartz:
    """
    Wet quartz rheological properties
    """
    def __init__(self):
        self.name = 'wet_quartz'
        self.pre_exponential_constant = 8.574e-28
        self.power_law_exponent = 4.0
        self.activation_energy = 222.0e3
        self.activation_volume = 0.0

class DryQuartz:
    """
    Dry quartz rheological properties
    """
    def __init__(self):
        self.name = 'dry_quartz'
        self.pre_exponential_constant = 0.0
        self.power_law_exponent = 0.0
        self.activation_energy = 0.0
        self.activation_volume = 0.0

class Basalt:
    """
    Basalt rheological properties
    """
    def __init__(self):
        self.name = 'basalt'
        self.pre_exponential_constant = 8.574e-28
        self.power_law_exponent = 4.0
        self.activation_energy = 222.0e3
        self.activation_volume = 0.0

class Plagioclase:
    """
    Plagioclase rheological properties (Shelton and Tullis, 1981)
    """
    def __init__(self):
        self.name = 'plagioclase' 
        self.pre_exponential_constant = 0.0
        self.power_law_exponent = 3.05
        self.activation_energy = 276.e3
        self.activation_volume = 0.0

class Air:
    """
    Air rheological properties
    """
    def __init__(self):
        self.name = 'air'
        self.pre_exponential_constant = 1.0e-18
        self.power_law_exponent = 1.0
        self.activation_energy = 0.0
        self.activation_volume = 0.0

###############################################################################################################################################
#Functions
###############################################################################################################################################

def find_nearest(array, value):
    '''Return the index in array nearest to a given value.
    
    Parameters
    ----------
    
    array: array_like
        1D array used to find the index
        
    value: float
        Value to be seached
    '''
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def read_params(fpath):
    '''
    Read Nx, Nz, Lx, Lz from param.txt
    '''
    with open(fpath+"param.txt","r") as f:
        line = f.readline()
        line = line.split() #split by space a string to a list of strings
        Nx = int(line[-1])
        
        line = f.readline()
        line = line.split()
        Nz = int(line[-1])

        line = f.readline()
        line = line.split()
        Lx = float(line[-1])

        line = f.readline()
        line = line.split()
        Lz = float(line[-1])

    return Nx, Nz, Lx, Lz

def read_data(prop, step, Nz, Nx, fpath):
    '''
    Read and reshape readed data according to parameters to return a (Nx, Nz) array
    '''
    
    #build filename
    filename = fpath + prop + "_" + str(step) + ".txt"

    data = np.loadtxt(filename, skiprows=2, unpack=True, comments="P")
    data = np.reshape(data, (Nz, Nx))
    
    return data

def calc_mean_temperaure_region(data, Nz, xx, begin, end):
    '''
    This funcition select a region in x direction in a 2D array and calculates the horizontal mean

    Parameters
    ----------

    data: `numpy.ndarray`

    Nz: int
        Number of points in Z direction

    xx: numpy.ndarray
        2D grid with x cordinates

    begin: float
        Start point

    end: float
        End point

    Returns
    -------
    arr: `numpy.ndarray`
        Array containing the horizontal mean of selected region
    '''

    x_region = (xx >= begin) & (xx <= end)
    Nx_aux = len(x_region[0][x_region[0]==True])
    data_sel = data[x_region].reshape(Nz, Nx_aux)
    data_sel_mean = np.mean(data_sel, axis=1)
    
    return data_sel_mean

###############################################################################################################################################
#Customizing matplotlib 

label_size=18
plt.rc('xtick', labelsize=label_size)
plt.rc('ytick', labelsize=label_size)

scenario_infos = ['SCENARIO INFOS:']
scenario_infos.append(' ')
scenario_infos.append('Name: ' + path[-1])

#Install the following package from (https://www.fabiocrameri.ch/colourmaps/) for inclusive color palletes
#or comment set crameri_colors as False

# crameri_colors=True
crameri_colors=False
if(crameri_colors):
    from cmcrameri import cm as cr
    def get_cycle(cmap, N=None, use_index="auto"):
        if isinstance(cmap, str):
            if use_index == "auto":
                if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                            'Dark2', 'Set1', 'Set2', 'Set3',
                            'tab10', 'tab20', 'tab20b', 'tab20c']:
                    use_index=True
                else:
                    use_index=False
            cmap = matplotlib.cm.get_cmap(cmap)
        if not N:
            N = cmap.N
        if use_index=="auto":
            if cmap.N > 100:
                use_index=False
            elif isinstance(cmap, LinearSegmentedColormap):
                use_index=False
            elif isinstance(cmap, ListedColormap):
                use_index=True
        if use_index:
            ind = np.arange(int(N)) % cmap.N
            return cycler("color",cmap(ind))
        else:
            colors = cmap(np.linspace(0,1,N))
            return cycler("color",colors)

    n_colors = 10
    # plt.rcParams["axes.prop_cycle"] = get_cycle(cr.romaO, n_colors)
    # plt.rcParams["axes.prop_cycle"] = get_cycle(cr.oslo, n_colors)
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cr.batlowKS(np.linspace(0, 1, 10)))
    #From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"])
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ['#88CCEE', '#44AA99', '#117733', '#332288', '#DDCC77', '#999933','#CC6677', '#882255', '#AA4499', '#585858'])
    


###############################################################################################################################################
#Setting the kind of tectonic scenario and number of cores
###############################################################################################################################################
# scenario_name = 'AR8'
# scenario_name = 'AR15'
# scenario_name = 'AR20'
# scenario_name = 'AR25'
# scenario_name = 'AR50'
# scenario_name = 'AR100'
scenario_name = 'AR150'

scenario_kind = 'accordion'

experiemnts = {
               'double_keel': 'Double Cratonic Keel',
               'accordion': 'Accordion',
               }

# ncores = 20
# ncores = 64
# ncores = 70
# ncores = 90
# # ncores = 128
# ncores = 192
# # ncores = 256
# # ncores = 384
# cores_per_node = 96


# #Estimating number of nodes needed according to number of cores
# nodes = (ncores + cores_per_node - 1) // cores_per_node #ceil division


#Rheological and Thermal parameters
# lower_crust_effective_viscosity_scale_factor = 1.0
lower_crust_effective_viscosity_scale_factor = 10.0

# homogeneous_mlit = True
# homogeneous_mlit = False

# velocity = 1.0 #cm/yr
# velocity = 2.0 #cm/yr
# velocity = 3.0 #cm/yr

# seed_in_litho = False
seed_in_litho = True

DeltaT = 0
# DeltaT = 290 # oC

# preset = True
preset = False

# selection_in_preset = True
selection_in_preset = False

# mean_litho = True
mean_litho = False

# high_kappa_in_asthenosphere = True
high_kappa_in_asthenosphere = False #default

splitted_local = local_path.split('/')
# path_to_preliminary_scenario = f"{'/'.join(splitted_local[:-1])}"
path_to_preliminary_scenario = f"/Volumes/Joao_Macedo/Doutorado/Silva_et_al_2026_Reconstruction_exhumed_mantle_data_and_scripts/scenarios"
scenario = f'{path_to_preliminary_scenario}/preliminary/' #Tp = 1350 oC

#Convergence criteria
# denok                            = 1.0e-15
denok                            = 2.0e-14 #was 1.0e-14
particles_per_element            = 100

#Surface constrains
sp_surface_tracking              = True
sp_surface_processes             = False
# sp_surface_processes             = True

#External inputs: bc velocity, velocity field, precipitation and
#climate change

variable_bcv                     = True
# variable_bcv                     = False
velocity_from_ascii              = True
ast_wind                         = False #True


#time constrains 
dt_max                           = 5.0e3
step_print                       = 100
if(variable_bcv == True):
    #first rifting phase
    if(scenario_name == 'AR8'):
        dt_rifting1 = 8.0
    if(scenario_name == 'AR15'):
            dt_rifting1 = 15.0
    if(scenario_name == 'AR20'):
        dt_rifting1 = 20.0
    if(scenario_name == 'AR25'):
        dt_rifting1 = 25.0
    if(scenario_name == 'AR50'):
        dt_rifting1 = 50.0
    if(scenario_name == 'AR100'):
        dt_rifting1 = 100.0
    if(scenario_name == 'AR150'):
        dt_rifting1 = 150.0
    
    ti_quiescence1 = 0 + dt_rifting1 #Myr

    #Time of quiescence1 to and start of convergence to close the ocean basin
    dt_quiescence1 = 20 #Myr
    tf_quiescence1 = ti_quiescence1 + dt_quiescence1

    #Closing the ocean basin over dt_rifting1 Myr and starting orogeny to begin the second quiescence phase
    dt_orogeny = 30.0 #Myr
    ti_quiescence2 = tf_quiescence1 + dt_rifting1 + dt_orogeny #Myr

    #second quiescence phase after orogeny
    dt_quiescence2 = 40.0 #Myr
    tf_quiescence2 = ti_quiescence2 + dt_quiescence2

    time_max = (tf_quiescence2)*1.0e6 #Myr to years
else:
    time_max = 40.0e6 #Myr to years

if(sp_surface_processes == True):
    precipitation_profile_from_ascii = True #False
    climate_change_from_ascii        = True #False
else:
    precipitation_profile_from_ascii = False
    climate_change_from_ascii        = False

# 
#step files
print_step_files                 = True
checkered = False
# checkered = True

#magmatism
# magmatism = 'off'
magmatism = 'on'
magmatism_extraction = 'on'

rheology_model = 19
#velocity bc
top_normal_velocity                 = 'fixed'         # ok
top_tangential_velocity             = 'free'         # ok
bot_normal_velocity                 = 'fixed'         # ok
bot_tangential_velocity             = 'free '         # ok
left_normal_velocity                = 'fixed'         # ok
# left_tangential_velocity            = 'free'         # ok
left_tangential_velocity            = 'fixed'         # ok
right_normal_velocity               = 'fixed'         # ok
right_tangential_velocity           = 'fixed'         # ok

# periodic_boundary = True
periodic_boundary = False
if(periodic_boundary == True):
    left_normal_velocity                = 'free'         # ok
    left_tangential_velocity            = 'free '         # ok
    right_normal_velocity               = 'free'         # ok
    right_tangential_velocity           = 'free'         # ok

#temperature bc
top_temperature                     = 'fixed'         # ok
bot_temperature                     = 'fixed'         # ok
left_temperature                    = 'fixed'          # ok
right_temperature                   = 'fixed'          # ok


###############################################################################################################################################
# Domain and interfaces
###############################################################################################################################################

Lx = 3500 * 1.0e3
# total model vertical extent (m)
Lz = 600 * 1.0e3
# number of points in horizontal direction
# Nx = 1751
Nx = 3501
# number of points in vertical direction
# Nz = 301
Nz = 601

# sediments = True
sediments = False

assimetric_cratons = True
# assimetric_cratons = False

if(sediments==True):
    # thickness of sticky air layer (m)
    thickness_air = 40 * 1.0e3
     #thickness of basalt layer (m)
    thickness_basalt = 0 * 1.0e3
    #thickness of sediments (m)
    thickness_sed = 3 * 1.0e3
    # thickness of decolement (m)
    thickness_decolement = 1 *1.0e3
    # thickness of upper crust (m)
    thickness_upper_crust = 21 * 1.0e3
    # thickness of lower crust (m)
    thickness_lower_crust = 10 * 1.0e3
    #Thickness of non cratonic lithosphere
    thickness_lithospheric_mantle = 85 * 1.0e3
    #Thickness of asthenosphere
    thickness_asthenosphere = Lz - (thickness_air + thickness_basalt + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle)
else:
    # thickness of sticky air layer (m)
    thickness_air = 40 * 1.0e3
    #thickness of basalt layer (m)
    thickness_basalt = 0 * 1.0e3
    # thickness of upper crust (m)
    thickness_upper_crust = 25 * 1.0e3
    # thickness of lower crust (m)
    thickness_lower_crust = 10 * 1.0e3
    #Thickness of non cratonic lithosphere
    thickness_lithospheric_mantle = 85 * 1.0e3
    #Thickness of asthenosphere
    thickness_asthenosphere = Lz - (thickness_air + thickness_basalt + thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle)

# total thickness of lithosphere (m)
if(sediments==True):
    thickness_litho = thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle #125 km - reference is the non-cratonic lithosphere
    thickness_astnc = Lz - (thickness_air + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle)
else:
    thickness_litho = thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle #125 km - reference is the non-cratonic lithosphere
    thickness_astnc = Lz - (thickness_air + thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle)
    thickness_asthenosphere = Lz - (thickness_air + thickness_basalt + thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle)

# seed depth bellow base of lower crust (m)
seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

x = np.linspace(0, Lx, Nx)
z = np.linspace(Lz, 0, Nz)
X, Z = np.meshgrid(x, z)
dz = Lz / (Nz - 1)


####################
# Setting layers  #
###################

if(sediments==True):
    thickness_lithosphere = thickness_basalt + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle
else:
    thickness_lithosphere = thickness_basalt + thickness_upper_crust + thickness_lower_crust + thickness_lithospheric_mantle

asthenosphere = MandyocLayer('asthenosphere', WetOlivine,
                            density=3378.0,
                            effective_viscosity_scale_factor=1.0,
                            radiogenic_heat_production=7.38e-12)

lithospheric_mantle = MandyocLayer('lithospheric_mantle', DryOlivine,
                                    density=3354.0,
                                    effective_viscosity_scale_factor=1.0,
                                    radiogenic_heat_production=9.0e-12,
                                    base_depth=thickness_air+thickness_upper_crust+thickness_lower_crust+thickness_lithospheric_mantle,
                                    Nx=Nx)

lower_crust = MandyocLayer('lower_crust', WetQuartz,
                            density=2800.0,
                            effective_viscosity_scale_factor=lower_crust_effective_viscosity_scale_factor,
                            radiogenic_heat_production=2.86e-10,
                            base_depth=thickness_air+thickness_upper_crust+thickness_lower_crust,
                            Nx=Nx) 

upper_crust = MandyocLayer('upper_crust', WetQuartz,
                            density=2700.0,
                            effective_viscosity_scale_factor=1.0,
                            radiogenic_heat_production=9.26e-10,
                            base_depth=thickness_air+thickness_upper_crust,
                            Nx=Nx)
if(sediments==True):
    sediments = MandyocLayer('sediments', WetQuartz,
                                density=2700.0,
                                effective_viscosity_scale_factor=1.0,
                                radiogenic_heat_production=1.25e-6 / 2700.0,
                                base_depth=thickness_air+thickness_sed,
                                Nx=Nx)

    decolement = MandyocLayer('decolement', Plagioclase,
                                density=2350.0,
                                effective_viscosity_scale_factor=0.1,
                                radiogenic_heat_production=1.25e-6 / 2700.0,
                                base_depth=thickness_air+thickness_sed+thickness_decolement,
                                Nx=Nx)

basalt = MandyocLayer('basalt', Basalt,
                        density=2900.0,
                        effective_viscosity_scale_factor=1.0,
                        radiogenic_heat_production=9.0e-11,
                        base_depth=thickness_air+thickness_basalt,
                        Nx=Nx)

air = MandyocLayer('air', Air,
                    density=1.0,
                    effective_viscosity_scale_factor=1.0,
                    radiogenic_heat_production=0.0,
                    base_depth=thickness_air,
                    Nx=Nx)

####################################
# Dealing with interface geometry  #
####################################

if(seed_in_litho):
    seed_depth = 6 * 1.0e3 #9 * 1.0e3 #original
    thickness_seed = 12 * 1.0e3

    seed_base = MandyocLayer('seed_base',
                                 DryOlivine,
                                 density=3354.0,
                                #  interface=np.ones(Nx) * (seed_depth + thickness_lower_crust + thickness_upper_crust + thickness_air),
                                 effective_viscosity_scale_factor=0.1,
                                 radiogenic_heat_production=9.0e-12,
                                 base_depth=lower_crust.base_depth+seed_depth,
                                 Nx=Nx)
    seed_top = MandyocLayer('seed_top',
                            DryOlivine,
                            density=3354.0,
                            # interface=np.ones(Nx) * (seed_depth + thickness_lower_crust + thickness_upper_crust + thickness_air),
                            effective_viscosity_scale_factor=0.1,
                            radiogenic_heat_production=9.0e-12,
                            base_depth=lower_crust.base_depth+seed_depth,
                            Nx=Nx) #0.8e-6 / 2800.0)
    
    # seed horizontal position (m)
    x_seed = Lx / 2.0
    # seed: number of points of horizontal extent
    n_seed = int(thickness_seed/1.0e3)

    seed_base.interface[int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)] = (
        seed_base.interface[
            int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
        ]
        + thickness_seed // 2
    )

    seed_top.interface[
        int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
    ] = (
        seed_top.interface[
            int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
        ]
        - thickness_seed // 2
    )

    if(sediments==True):
        layers = [asthenosphere, lithospheric_mantle, seed_base, seed_top, lower_crust, upper_crust, decolement, sediments, basalt, air]
    else:
        layers = [asthenosphere, lithospheric_mantle, seed_base, seed_top, lower_crust, upper_crust, basalt, air]
else:
    if(sediments==True):
        layers = [asthenosphere, lithospheric_mantle, lower_crust, upper_crust, decolement, sediments, basalt, air]
    else:
        layers = [asthenosphere, lithospheric_mantle, lower_crust, upper_crust, basalt, air]

##################################################
# Save interfaces.txt to be used in Mandyoc code #
##################################################

#Build layer_properties according to the order of the stack_layers
rheological_symbols = ['C', 'rho', 'H', 'A', 'n', 'Q', 'V']
rheological_properties = ['effective_viscosity_scale_factor',
                          'density',
                          'radiogenic_heat_production',
                          'pre_exponential_constant',
                          'power_law_exponent',
                          'activation_energy',
                          'activation_volume']

to_save = []
for symbol, prop in zip(rheological_symbols, rheological_properties):
    to_save.append(f"{symbol} {' '.join([str(layer.__dict__[prop]) for layer in layers])}")

with open("interfaces.txt", "w") as f:

    for line in to_save:
        if len(line):
            f.write(f"{' '.join(line.split())}\n")

    # layer interfaces
    data = -1 * np.array(tuple(layer.interface for layer in layers[1::])).T #excludin asthenosphere interface
    np.savetxt(f, data, fmt="%.1f")

##############################################################################
# Parameters file
##############################################################################
params = f"""
nx = {Nx}
nz = {Nz}
lx = {Lx}
lz = {Lz}
# Simulation options
multigrid                           = 1             # ok -> soon to be on the command line only
solver                              = direct        # default is direct [direct/iterative]
denok                               = {denok}       # default is 1.0E-4
particles_per_element               = {particles_per_element}          # default is 81
surface_particles_per_element       = 40           # default is 2
particles_perturb_factor            = 0.7           # default is 0.5 [values are between 0 and 1]
rtol                                = 1.0e-7        # the absolute size of the residual norm (relevant only for iterative methods), default is 1.0E-5
RK4                                 = Euler         # default is Euler [Euler/Runge-Kutta]
Xi_min                              = 1.0e-5       # default is 1.0E-14
random_initial_strain               = 0.3           # default is 0.0
pressure_const                      = -1.0          # default is -1.0 (not used) - useful only in horizontal 2D models
initial_dynamic_range               = True         # default is False [True/False]
periodic_boundary                   = False         # default is False [True/False]
high_kappa_in_asthenosphere         = {high_kappa_in_asthenosphere}         # default is False [True/False]
K_fluvial                           = 2.0e-7        # default is 2.0E-7
m_fluvial                           = 1.0           # default is 1.0
sea_level                           = 0.0           # default is 0.0
basal_heat                          = 0.0          # default is -1.0
# Surface processes
sp_surface_tracking                 = {sp_surface_tracking}         # default is False [True/False]
sp_surface_processes                = {sp_surface_processes}        # default is False [True/False]
plot_sediment                       = False         # default is False [True/False]
a2l                                 = True          # default is True [True/False]
free_surface_stab                   = True         # default is True [True/False]
theta_FSSA                          = 0.5          # default is 0.5 (only relevant when free_surface_stab = True)
# Time constrains
step_max                            = 800000        # Maximum time-step of the simulation
time_max                            = {time_max}    # Maximum time of the simulation [years]
dt_max                              = {dt_max}      # Maximum time between steps of the simulation [years]
step_print                          = {step_print}  # Make file every <step_print>
sub_division_time_step              = 0.5           # default is 1.0
initial_print_step                  = 0             # default is 0
initial_print_max_time              = 1.0e6         # default is 1.0E6 [years]
# Viscosity
viscosity_reference                 = 1.0e26        # Reference viscosity [Pa.s]
viscosity_max                       = 1.0e25        # Maximum viscosity [Pa.s]
viscosity_min                       = 1.0e18        # Minimum viscosity [Pa.s]
viscosity_per_element               = constant      # default is variable [constant/variable]
viscosity_mean_method               = arithmetic      # default is harmonic [harmonic/arithmetic]
viscosity_dependence                = pressure      # default is depth [pressure/depth]
# External ASCII inputs/outputs
interfaces_from_ascii               = True          # default is False [True/False]
n_interfaces                        = {len(layers)-1}           # Number of interfaces int the interfaces.txt file
variable_bcv                        = {variable_bcv} #False         # default is False [True/False]
temperature_from_ascii              = True         # default is False [True/False]
velocity_from_ascii                 = {velocity_from_ascii} #False      # default is False [True/False]
binary_output                       = False         # default is False [True/False]
sticky_blanket_air                  = True         # default is False [True/False]
# precipitation_profile_from_ascii    = {precipitation_profile_from_ascii}         # default is False [True/False]
# climate_change_from_ascii           = {climate_change_from_ascii}         # default is False [True/False]
print_step_files                    = {print_step_files}          # default is True [True/False]
checkered                           = {checkered}         # Print one element in the print_step_files (default is False [True/False])
geoq                                = on            # ok
geoq_fac                            = 100.0           # ok
# HDF5
output_hdf5 = True
# Snapshot
snapshot_interval = 1.0e5
snapshot_files = 2
# Physical parameters
temperature_difference              = 1500.         # ok
thermal_expansion_coefficient       = 3.28e-5       # ok
thermal_diffusivity_coefficient     = 1.0e-6 #0.75e-6       #default is 1.0e-6        # ok
gravity_acceleration                = 10.0          # ok
density_mantle                      = 3300.         # ok
external_heat                       = 0.0e-12       # ok
heat_capacity                       = 1250.         # ok #default is 1250
non_linear_method                   = on            # ok
adiabatic_component                 = on            # ok
radiogenic_component                = on            # ok
magmatism                           = {magmatism}           # ok
magmatism_extraction                = {magmatism_extraction}
export_lithology = True
magmatic_layer = 6
# Velocity boundary conditions
top_normal_velocity                 = fixed         # ok
top_tangential_velocity             = free          # ok
bot_normal_velocity                 = fixed         # ok
bot_tangential_velocity             = free          # ok
left_normal_velocity                = {left_normal_velocity}         # ok
left_tangential_velocity            = {left_tangential_velocity}          # ok
right_normal_velocity               = {right_normal_velocity}         # ok
right_tangential_velocity           = {right_tangential_velocity}         # ok
surface_velocity                    = 0.0e-2        # ok
multi_velocity                      = False         # default is False [True/False]
# Temperature boundary conditions
top_temperature                     = {top_temperature}         # ok
bot_temperature                     = {bot_temperature}         # ok
left_temperature                    = {left_temperature}         # ok
right_temperature                   = {right_temperature}         # ok
rheology_model                      = {rheology_model}             # ok
T_initial                           = 3             # ok
"""
# Create the parameter file
with open("param.txt", "w") as f:
    for line in params.split("\n"):
        line = line.strip()
        if len(line):
            f.write(" ".join(line.split()) + "\n")


##############################################################################
# Initial temperature field
##############################################################################

if(preset == False):
    T = 1300 * (z - thickness_air) / (thickness_lithosphere)  # Temperature of 1300 isotherm bellow the lithosphere

    ccapacity = 1250*1.0 #937.5=75% #J/kg/K? #DEFAULT

    TP = 1262 #mantle potential temperature

    Ta = (TP / np.exp(-10 * 3.28e-5 * (z - thickness_air) / ccapacity)) + DeltaT
    # Ta = 1262 / np.exp(-10 * 3.28e-5 * (z - thickness_air) / ccapacity)steady s

    T[T < 0.0] = 0.0
    cond1 = Ta<T #VICTOR
    T[T > Ta] = Ta[T > Ta] #apply the temperature of asthenosphere Ta where temperature T is greater than Ta, 

    kappa = 1.0e-6 #thermal diffusivity

    H = np.zeros_like(T)

    cond = (z >= thickness_air) & (z < thickness_upper_crust + thickness_air)  # upper crust
    H[cond] = upper_crust.radiogenic_heat_production

    cond = (z >= thickness_upper_crust + thickness_air) & (
        z < thickness_lower_crust + thickness_upper_crust + thickness_air
    )  # lower crust
    H[cond] = lower_crust.radiogenic_heat_production

    Taux = np.copy(T)
    t = 0
    dt = 5000
    dt_sec = dt * 365 * 24 * 3600
    # cond = (z > thickness_air + thickness_lithosphere) | (T == 0)  # (T > 1300) | (T == 0) #OLD
    cond = cond1 | (T == 0)  # (T > 1300) | (T == 0) #VICTOR
    dz = Lz / (Nz - 1)

    
    while t < 500.0e6:
        T[1:-1] += (
            kappa * dt_sec * ((T[2:] + T[:-2] - 2 * T[1:-1]) / dz ** 2)
            + H[1:-1] * dt_sec / ccapacity
        )
        T[cond] = Taux[cond]
        t = t + dt
    
    T = np.ones_like(X) * T[:, None] #(Nz, Nx)

    # print('shape T: ', np.shape(T))

    # Save the initial temperature file
    np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Nz)), header="T1\nT2\nT3\nT4")

else:
    dz = Lz / (Nz - 1)

    from_dataset = True
    # from_dataset = False

    if(from_dataset == True):

        # external_media = 'Joao_Macedo'
        # fpath = f"{machine_path}/{external_media}{scenario}"

        dataset = xr.open_dataset(f"{scenario}/_output_temperature.nc")
        
        Nx_aux = int(dataset.nx)
        Nz_aux = int(dataset.nz)
        Lx_aux = float(dataset.lx)
        Lz_aux = float(dataset.lz)

        x_aux = np.linspace(0, Lx_aux, Nx_aux)
        z_aux = np.linspace(Lz_aux, 0, Nz_aux)
        xx_aux, zz_aux  = np.meshgrid(x_aux, z_aux)

        time = dataset.time[-1]
        Datai = dataset.temperature[-1].values.T
    else:
        fpath = f"{scenario}/"
        Nx_aux, Nz_aux, Lx_aux, Lz_aux = read_params(fpath)

        x_aux = np.linspace(0, Lx_aux, Nx_aux)
        z_aux = np.linspace(Lz_aux, 0, Nz_aux)
        xx_aux, zz_aux  = np.meshgrid(x_aux, z_aux)

        steps = sorted(glob.glob(fpath+"time_*.txt"), key=os.path.getmtime)
        step_final = int(steps[-1].split('/')[-1][5:-4]) #step of final thermal structure
        
        time_fname = fpath + 'time_' + str(step_final) + '.txt'
        time = np.loadtxt(time_fname, usecols=2, max_rows=1)

        Datai = read_data('temperature', step_final, Nz_aux, Nx_aux, fpath) #(read final thermal structure (Nz, Nx)
    

    #Setting procedure with external temperature field. Choose between:
        ##Use the horizontal mean of temperature from final step of used scenario (horizontal_mean)
        ##or
        ##Use the original thermal state used as input interpolated on new grid Nx x Nz (interp2d)


    interp_method = 'horizontal_mean' #using interp1d
    # interp_method = 'interp2d'

    if(interp_method == 'horizontal_mean'):
        datai_mean = np.mean(Datai, axis=1) #horizontal mean

        f = interp1d(z_aux, datai_mean) #funcion to interpolate the temperature field
        datai_mean_interp = f(z) #applying the function to obtain the temperature field to the new mesh

        zcond = z <= 40.0e3
        datai_mean_interp[datai_mean_interp <= 1.0e-7] = 0.0 #dealing with <=0 values inherited from interpolation
        datai_mean_interp[zcond] = 0.0

        T = np.zeros((Nx, Nz)) #(Nx, Nz) = transpose of original shape (Nz, Nx)
        
        for i in range(Nx): #len(Nx) 
            T[i, :] = datai_mean_interp

        T = T.T #(Nz,Nx): transpose T to plot below
        # print('shape T: ', np.shape(T))
    
    else:
        interp_kind = 'linear'
        # interp_kind = 'cubic'
        # interp_kind = 'quintic'

        f = interp2d(x_aux, z_aux, Datai, kind=interp_kind)
        temper_interp = f(x, z) #(Nz, Nx)
        temper_interp[temper_interp <= 1.0e-7] = 0.0 #dealing with <=0 values inherited from interpolation
        
        #Setting temperature on vertical boundaries. Choose between:
        ##Use the mean temperature from final step of used scenario (mean)
        ##or
        ##Use the original thermal state used as input interpolated on new Nz (original)
        bound = 'mean'
        # bound = 'original'

        if(bound == 'mean'):
            #Calc horizontal mean from interpolated field
            temper_interp_mean = np.mean(temper_interp, axis=1)
            zcond = z >= 660.0e3 #temperature field is from bottom to top
            temper_interp_mean[zcond] = 0.0
            
            if(mean_litho==True):

                if(thickness_lithosphere == 80.0e3):
                    zcond1 = (z >= 660.0e3-thickness_lithosphere) & (z < 660.0e3)
                else:
                    zcond1 = (z >= 560.0e3) & (z < 660.0e3)
                temper_interp = temper_interp.T #Change to (Nx, Nz)
                
                for i in range(Nx): #len(Nx)
                    temper_interp[i][zcond1] = temper_interp_mean[zcond1]

                temper_interp = temper_interp.T #Return to (Nz, Nx)

            #Apply horizontal mean to vertical boundaries
            for i in range(Nz):
                temper_interp[i][0] = temper_interp_mean[i]
                temper_interp[i][-1] = temper_interp_mean[i]

        else:
            #Cat the initial thermal state from scenario
            step_initial = int(steps[0].split('/')[-1][5:-4])
            time_fname = fpath + 'time_' + str(step_initial) + '.txt'
            time = np.loadtxt(time_fname, usecols=2, max_rows=1)
            T0i = read_data('temperature', step_initial, Nz_aux, Nx_aux, fpath)

            #interpolate in new grid
            T0 = T0i[:, 0]
            f = interp1d(z_aux, T0)
            T0_interp = f(z)
            T0_interp[T0_interp<=1.0e-7] = 0.0
            T0_interp = T0_interp[::-1]
            
            #apply to boundaries
            for i in range(Nz):
                temper_interp[i][0] = T0_interp[i]
                temper_interp[i][-1] = T0_interp[i]
         
        T = temper_interp[::-1]

    np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Nz)), header="T1\nT2\nT3\nT4")

##############################################################################
# Boundary condition - velocity
##############################################################################
velocity = 1.0 #cm/yr

if(velocity_from_ascii == True):

    fac_air = 10.0e3
    #1.0 cm/yr
    # vL = 0.005 / (365 * 24 * 3600)  # m/s
    
    # 0.5 cm/year
    # vL = 0.0025 / (365 * 24 * 3600)  # m/s
    
    # 0.25 cm/year
    # vL = 0.00125 / (365 * 24 * 3600)  # m/s

    vL = (0.5*velocity/100) / (365 * 24 * 3600)  # m/s

    h_v_const = thickness_lithosphere + 20.0e3  #thickness with constant velocity 
    ha = Lz - thickness_air - h_v_const  # difference

    vR = 2 * vL * (h_v_const + fac_air + ha) / ha  # this is to ensure integral equals zero

    VX = np.zeros_like(X)
    cond = (Z > h_v_const + thickness_air) & (X == 0)
    VX[cond] = vR * (Z[cond] - h_v_const - thickness_air) / ha

    cond = (Z > h_v_const + thickness_air) & (X == Lx)
    VX[cond] = -vR * (Z[cond] - h_v_const - thickness_air) / ha

    cond = X == Lx
    VX[cond] += +2 * vL

    cond = Z <= thickness_air - fac_air
    VX[cond] = 0

    # print(np.sum(VX))

    v0 = VX[(X == 0)]
    vf = VX[(X == Lx)]
    sv0 = np.sum(v0[1:-1]) + (v0[0] + v0[-1]) / 2.0
    svf = np.sum(vf[1:-1]) + (vf[0] + vf[-1]) / 2.0
    # print(sv0, svf, svf - sv0)

    diff = (svf - sv0) * dz

    vv = -diff / Lx
    # print(vv, diff, svf, sv0, dz, Lx)

    VZ = np.zeros_like(X)

    cond = Z == 0
    VZ[cond] = vv
    #save bc to plot arraows in numerical setup
    vels_bc = np.array([v0, vf])
    vz0 = VZ[(z == 0)]

    np.savetxt("vel_bc.txt", vels_bc.T)
    np.savetxt("velz_bc.txt", vz0.T)

    VVX = np.copy(np.reshape(VX, Nx * Nz))
    VVZ = np.copy(np.reshape(VZ, Nx * Nz))

    v = np.zeros((2, Nx * Nz))

    v[0, :] = VVX
    v[1, :] = VVZ

    v = np.reshape(v.T, (np.size(v)))

    # Create the initial velocity file
    np.savetxt("input_velocity_0.txt", v, header="v1\nv2\nv3\nv4")

# print(np.sum(v0))

VVX = np.copy(np.reshape(VX, Nx * Nz))
VVZ = np.copy(np.reshape(VZ, Nx * Nz))

v = np.zeros((2, Nx * Nz))

v[0, :] = VVX
v[1, :] = VVZ

v = np.reshape(v.T, (np.size(v)))

# Create the initial velocity file
np.savetxt("input_velocity_0.txt", v, header="v1\nv2\nv3\nv4")

if(variable_bcv == True):

    var_bcv = f""" 3
                    {ti_quiescence1} 0.01
                    {tf_quiescence1} -100.0
                    {ti_quiescence2} 0.01
                    """

    # Create the parameter file
    with open("scale_bcv.txt", "w") as f:
        for line in var_bcv.split("\n"):
            line = line.strip()
            if len(line):
                f.write(" ".join(line.split()) + "\n")
else:
    time_max = 120.0e6




if(sp_surface_processes == True):
    if(climate_change_from_ascii == True):
        #When climate effects will start to act - scaling to 1
        # climate = f'''
        #         2
        #         0 0.0
        #         10 0.02
        #     '''

        climate = f'''
                2
                0 0.0
                120 0.02
            '''

        with open('climate.txt', 'w') as f:
            for line in climate.split('\n'):
                line = line.strip()
                if len(line):
                    f.write(' '.join(line.split()) + '\n')

    if(precipitation_profile_from_ascii ==True):
        #Creating precipitation profile

        prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(1))**6) #Lx km
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/8)**6) #original
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(8*2))**6) #100 km
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(8*4))**6) #50 

        np.savetxt("precipitation.txt", prec, fmt="%.8f")

######################################################################################
#Creating numerical setup figure to visualize the initial configuration of the model #
######################################################################################

plt.close()
fig, axs = plt.subplots(1, 1, figsize = (14, 6))
ylimplot = [-Lz/1000+thickness_air/1000, 0+thickness_air/1000]
#plot scenario layers
#layers colour scheme
cr = 255.
color_air = "xkcd:white"
color_basalt = "xkcd:red"
color_sed = (241./cr,184./cr,68./cr)
color_dec = (137./cr,81./cr,151./cr)
color_uc = (228./cr,156./cr,124./cr)
color_lc = (240./cr,209./cr,188./cr)
color_lit = (155./cr,194./cr,155./cr)
color_ast = (207./cr,226./cr,205./cr)

if(sediments==True):
    colors = {'air': color_basalt,
        'basalt': color_sed,
        'sediments': color_dec,
        'decolement':color_uc,
        'upper_crust': color_lc,
        'seed_top': color_lc,
        'seed_base': color_lc,
        'lower_crust': color_lit,
        'lithospheric_mantle': color_ast,
    }
else:
    # colors = {'air': color_uc,
    #           'basalt': color_basalt,
    #           'upper_crust': color_lc,
    #           'seed_top': color_lc,
    #           'seed_base': color_lc,
    #           'lower_crust': color_lit,
    #           'litho_nc': color_ast,}
    # labels = {
    #     'air': 'Upper crust',
    #     'upper_crust': 'Lower crust',
    #     'lower_crust': 'Lithospheric mantle',
    #     'litho_nc': 'Asthenosphere',#'Upper cratonic lithospheric mantle',
    # }

    colors = {'air': color_uc,
              'basalt': color_uc,
              'upper_crust': color_lc,
              'seed_top': color_lit,
              'seed_base': color_lit,
              'lower_crust': color_lit,
              'lithospheric_mantle': color_ast,}
    labels = {
        'air': 'Basalt',
        'basalt':'Upper crust',
        'upper_crust': 'Lower crust',
        'lower_crust': 'Lithospheric mantle',
        'lithospheric_mantle': 'Asthenosphere',#'Upper cratonic lithospheric mantle',
    }
layers_aux = layers[::-1]

for layer in layers_aux[:-1:]:
    layer_name = str(layer.layer_label)
    # print(layer_name)
    if(layer_name == 'basalt' or layer_name == 'seed_top'):
        continue

    if(layer_name == 'seed_base'):
        plt.plot(x_seed/1.0E3, (-seed_depth-thickness_air)/1.0E3, 'x', color='xkcd:black', lw=1.0)
    else:
        axs.plot(x/1.0E3, (-layer.interface+thickness_air)/1.0E3, color='xkcd:black')
        layer_name = str(layer.layer_label)
        # axs.plot(x/1.0E3, (-layer.interface+thickness_air)/1.0E3, label=layer_name, lw=2)
        axs.fill_between(x/1000, -layer.interface/1000+thickness_air/1000, -Lz/1000+thickness_air/1000, color=colors[layer_name])#, label=labels[label])

# for interface in list(interfaces.items())[::-1]:
#     label, layer = interface[0], interface[1]
#     # if(label!='litho_crat_up'):
#     #     axs.plot(x/1000, (-layer)/1000+thickness_air/1000, color='k', lw=0.5)
#     # if(label=='seed_base' or label=='seed_top'):
#     #     axs.plot(x/1000, (-layer)/1000+thickness_air/1000, color='k', lw=0.5)
#     if(label=='seed_top'):
#         axs.plot(x_seed/1000, -(thickness_upper_crust + thickness_lower_crust - seed_depth)/1000, 'x', color='k', lw=1.0)

#     if(label == 'seed_top' or label == 'seed_base'):
#         label = 'lower_crust'
#         continue
#     if(label == 'litho_crat_up'):
#         continue

    # axs.fill_between(x/1000, -layer/1000+thickness_air/1000, -Lz/1000+thickness_air/1000, color=colors[label])#, label=labels[label])

dx = Lx/(Nx-1)
axs.set_xlim(0, Lx/1000)
axs.set_xticks([])
axs.set_yticks([])
axs.set_ylim(ylimplot)
axs.set_xlabel(f"Distance = {Lx/1000:.0f} km\nResolution = {dx/1000:.0f} km", fontsize=16)
axs.set_ylabel(f"Depth = {Lz/1000:.0f} km\nResolution = {dz/1000:.0f} km", fontsize=16)

#plotting ghost points to create a legend to the layers
colors_legend = {'air': color_air,
                 'sediments': color_sed,
                 'decolement':color_dec,
                 'upper_crust': color_uc,
                 'seed_top': color_lc,
                 'seed_base': color_lc,
                 'lower_crust': color_lc,
                 'lithospheric_mantle': color_lit,
                 'asthenosphere': color_ast,}

if(sediments==True):
    labels_legend = {
        'air': f"Sticky air\n{air.effective_viscosity_scale_factor:.0f} x air\n"+fr"$\rho$ = {air.density:.0f} kg/m³"+f"\n$h$={thickness_air/1.0E3:.0f} km",
        'sediments': f"Sediments\n{sediments.effective_viscosity_scale_factor:.0f} x wet quartz\n"+fr"$\rho$ = {sediments.density:.0f} kg/m³"+f"\nh={thickness_sed/1.0E3:.0f} km",
        'decolement': f"Decolement\n{decolement.effective_viscosity_scale_factor:.1f} x wet quartz\n"+fr"$\rho$ = {decolement.density:.0f} kg/m³"+f"\nh={thickness_decolement/1.0E3:.0f} km",
        'upper_crust': f"Upper crust\n{upper_crust.effective_viscosity_scale_factor:.0f} x wet quartz\n"+fr"$\rho$ = {upper_crust.density:.0f} kg/m³"+f"\nh={thickness_upper_crust/1.0E3:.0f} km",
        'lower_crust': f"Lower crust\n{lower_crust.effective_viscosity_scale_factor:.0f} x wet quartz\n"+fr"$\rho$ = {lower_crust.density:.0f} kg/m³"+f"\nh={thickness_lower_crust/1.0E3:.0f} km",
        'lithospheric_mantle': f"Lithospheric\nmantle\n{lithospheric_mantle.effective_viscosity_scale_factor:.0f} x dry olivine\n"+fr"$\rho$ = {lithospheric_mantle.density:.0f} kg/m³"+f"\nh={thickness_lithospheric_mantle/1.0E3:.0f} km",
        'asthenosphere': f"Asthenosphere\n{asthenosphere.effective_viscosity_scale_factor:.0f} x wet olivine\n"+fr"$\rho$ = {asthenosphere.density:.0f} kg/m³"+f"\nh={thickness_asthenosphere/1.0E3:.0f} km"}
else:
    labels_legend = {
        'air': f"Sticky air\n{air.effective_viscosity_scale_factor:.0f} x air\n"+fr"$\rho$ = {air.density:.0f} kg/m³"+f"\n$h$ = {thickness_air/1.0E3:.0f} km",
        'upper_crust': f"Upper crust\n{upper_crust.effective_viscosity_scale_factor:.0f} x wet quartz\n"+fr"$\rho$ = {upper_crust.density:.0f} kg/m³"+f"\n$h$ = {thickness_upper_crust/1.0E3:.0f} km",
        'lower_crust': f"Lower crust\n{lower_crust.effective_viscosity_scale_factor:.0f} x wet quartz\n"+fr"$\rho$ = {lower_crust.density:.0f} kg/m³"+f"\n$h$ = {thickness_lower_crust/1.0E3:.0f} km",
        'lithospheric_mantle': f"Lithospheric\nmantle\n{lithospheric_mantle.effective_viscosity_scale_factor:.0f} x dry olivine\n"+fr"$\rho$ = {lithospheric_mantle.density:.0f} kg/m³"+f"\n$h$ = {thickness_lithospheric_mantle/1.0E3:.0f} km",
        'asthenosphere': f"Asthenosphere\n{asthenosphere.effective_viscosity_scale_factor:.0f} x wet olivine\n"+fr"$\rho$ = {asthenosphere.density:.0f} kg/m³"+f"\n$h$ = {thickness_asthenosphere/1.0E3:.0f} km"}

legend_elements = []
for key in labels_legend.keys():
    legend_elements.append(Patch(facecolor=colors_legend[key], edgecolor='black', label=labels_legend[key]))

fig.subplots_adjust(bottom=0.25)

leg = axs.legend(handles=legend_elements,
    ncol=len(legend_elements), 
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.45), 
    frameon=False,
    title='Layers properties',
    title_fontsize=15,
    fontsize=8,
    columnspacing=1.0,
)

#Indicating weak seed position
xpos_seed = x_seed/Lx
correction = 0.90
if(sediments==True):
    ypos_seed = correction*(1-(thickness_air + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust)/Lz)
else:
    ypos_seed = correction*(1-(thickness_air + thickness_upper_crust + thickness_lower_crust)/Lz)
axs.text(xpos_seed, ypos_seed, f'weak seed\n{thickness_seed/1000:.0f}x{thickness_seed/1000:.0f} km²', color='k', fontsize=12, ha='center', va='center', transform=axs.transAxes)


#Temperature profiles
idx_center = int((Nx-1)/2) 
axt = axs.inset_axes((0.205,
                      0,
                      0.18,
                      1))

axt.plot(T[:, idx_center], (-z + thickness_air) / 1.0e3, "-r")# label=r'T$_{\mathrm{non-cratonic}}$')
axt.grid(visible=True, axis='x',which='both',ls='--',color='red',alpha=0.3)
axt.set_ylim(ylimplot)
axt.set_yticks([])
axt.set_xticks(np.linspace(0,1800,7))
axt.patch.set_alpha(0)
axt.xaxis.set_ticks_position('top')
axt.set_yticks([])
axt.tick_params(labelsize=8)
axt.xaxis.label.set_color('red')
axt.tick_params(axis='x', colors='k')
axt.spines['left'].set_visible(False)
axt.spines['right'].set_visible(False)
axt.set_title('Temperature [°C]',color='k')
# axt.legend(loc='lower left', fontsize=10, framealpha=0.9)
#axt.spines['bottom'].set_visible(False)

################################
#    Yield Strength Envelope   #
################################

Q = np.zeros_like(z)
A = np.zeros_like(z)
n = np.zeros_like(z)
V = np.zeros_like(z)
C = np.zeros_like(z)
rho = np.zeros_like(z)

zaux = z
if(sediments==True):
    sa = zaux < thickness_air
    sed = (zaux>thickness_air) & (zaux<thickness_air+thickness_sed)
    dec = (zaux>thickness_air+thickness_sed) & (zaux<thickness_air+thickness_sed+thickness_decolement)
    uc =  (zaux>=thickness_air+thickness_sed+thickness_decolement) & (zaux<thickness_air+thickness_sed+thickness_decolement+thickness_upper_crust)
    lc =  (zaux>=thickness_air+thickness_sed+thickness_decolement+thickness_upper_crust) & (zaux<thickness_air+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust)
    lm =  (zaux>=thickness_air+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust) & (zaux<=thickness_air+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust+thickness_lithospheric_mantle)
    astnc = zaux>thickness_air+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust+thickness_lithospheric_mantle 
    
    #non cratonic rheological properties
    C[sa] = air.effective_viscosity_scale_factor
    C[sed] = sediments.effective_viscosity_scale_factor
    C[dec] = decolement.effective_viscosity_scale_factor
    C[uc] = upper_crust.effective_viscosity_scale_factor
    C[lc] = lower_crust.effective_viscosity_scale_factor
    C[lm] = lithospheric_mantle.effective_viscosity_scale_factor
    C[astnc] = asthenosphere.effective_viscosity_scale_factor

    rho[sa] = air.density
    rho[sed] = sediments.density
    rho[dec] = decolement.density
    rho[uc] = upper_crust.density
    rho[lc] = lower_crust.density
    rho[lm] = lithospheric_mantle.density
    rho[astnc] = asthenosphere.density

    A[sa] = air.pre_exponential_constant
    A[sed] = sediments.pre_exponential_constant
    A[dec] = decolement.pre_exponential_constant
    A[uc] = upper_crust.pre_exponential_constant
    A[lc] = lower_crust.pre_exponential_constant
    A[lm] = lithospheric_mantle.pre_exponential_constant
    A[astnc] = asthenosphere.pre_exponential_constant

    n[sa] = air.power_law_exponent
    n[sed] = sediments.power_law_exponent
    n[dec] = decolement.power_law_exponent
    n[uc] = upper_crust.power_law_exponent
    n[lc] = lower_crust.power_law_exponent
    n[lm] = lithospheric_mantle.power_law_exponent
    n[astnc] = asthenosphere.power_law_exponent

    Q[sa] = air.activation_energy
    Q[sed] = sediments.activation_energy
    Q[dec] = decolement.activation_energy
    Q[uc] = upper_crust.activation_energy
    Q[lc] = lower_crust.activation_energy
    Q[lm] = lithospheric_mantle.activation_energy
    Q[astnc] = asthenosphere.activation_energy

    V[sa] = air.activation_volume
    V[sed] = sediments.activation_volume
    V[dec] = decolement.activation_volume
    V[uc] = upper_crust.activation_volume
    V[lc] = lower_crust.activation_volume
    V[lm] = lithospheric_mantle.activation_volume
    V[astnc] = asthenosphere.activation_volume

else:
    sa = zaux < thickness_air
    uc = (zaux>=thickness_air) & (zaux<thickness_air+thickness_upper_crust)
    lc = (zaux>=thickness_air+thickness_upper_crust) & (zaux<thickness_air+thickness_upper_crust+thickness_lower_crust)
    lm = (zaux>=thickness_air+thickness_upper_crust+thickness_lower_crust) & (zaux<=thickness_air+thickness_upper_crust+thickness_lower_crust+thickness_lithospheric_mantle)
    ast = zaux>thickness_air+thickness_upper_crust+thickness_lower_crust+thickness_lithospheric_mantle
    
    #non cratonic rheological properties
    C[sa] = air.effective_viscosity_scale_factor
    C[uc] = upper_crust.effective_viscosity_scale_factor
    C[lc] = lower_crust.effective_viscosity_scale_factor
    C[lm] = lithospheric_mantle.effective_viscosity_scale_factor
    C[ast] = asthenosphere.effective_viscosity_scale_factor

    rho[sa] = air.density
    rho[uc] = upper_crust.density
    rho[lc] = lower_crust.density
    rho[lm] = lithospheric_mantle.density
    rho[ast] = asthenosphere.density

    A[sa] = air.pre_exponential_constant
    A[uc] = upper_crust.pre_exponential_constant
    A[lc] = lower_crust.pre_exponential_constant
    A[lm] = lithospheric_mantle.pre_exponential_constant
    A[ast] = asthenosphere.pre_exponential_constant

    n[sa] = air.power_law_exponent
    n[uc] = upper_crust.power_law_exponent
    n[lc] = lower_crust.power_law_exponent
    n[lm] = lithospheric_mantle.power_law_exponent
    n[ast] = asthenosphere.power_law_exponent

    Q[sa] = air.activation_energy
    Q[uc] = upper_crust.activation_energy
    Q[lc] = lower_crust.activation_energy
    Q[lm] = lithospheric_mantle.activation_energy
    Q[ast] = asthenosphere.activation_energy

    V[sa] = air.activation_volume
    V[uc] = upper_crust.activation_volume
    V[lc] = lower_crust.activation_volume
    V[lm] = lithospheric_mantle.activation_volume
    V[ast] = asthenosphere.activation_volume


sr = 1.0E-15 #strain rate - s-1
# sr = 1.0E-14
R = 8.314 #gas constant - J K−1 mol−1
g = 10.0

P = rho[::-1].cumsum()[::-1]*g*dz

phi = 2.0*np.pi/180.0
c0 = 4.0E6

sigmanc_min = c0 * np.cos(phi) + P * np.sin(phi)

phi = 15.0*np.pi/180.0
c0 = 20.0E6
sigma_max = c0 * np.cos(phi) + P * np.sin(phi)

TK = T[:, 0] + 273

visc = C * A**(-1./n) * sr**((1.0-n)/n)*np.exp((Q + V*P)/(n*R*TK))
sigma_v = visc * sr
cond = sigma_v>sigma_max
sigma_v[cond]=sigma_max[cond]

axsg = axs.inset_axes((0.605,
                       0,
                       0.13,
                       1))
if(sediments==True):
    axsg.plot(sigma_v/1e9,-(z-thickness_air)/1e3,'r', label=f'Non-cratonic')
    # axsg.plot(sigmanc_min/1e9,-(z-t_sa)/1e3,'k--',lw=0.8)
    # axsg.plot(sigmac_v/1e9,-(z-thickness_air)/1e3,'k', label=f'Cratonic')
else:
    axsg.plot(sigma_v/1e9,-(z-thickness_air)/1e3,'r', label=f'Non-cratonic')
    # axsg.plot(sigmanc_min/1e9,-(z-t_sa)/1e3,'k--',lw=0.8)
    # axsg.plot(sigmac_v/1e9,-(z-thickness_air)/1e3,'k', label=f'Cratonic')
    # axsg.plot(sigmac_min/1e9,-(z-t_sa)/1e3,'k--',lw=0.8)

axsg.grid(visible=True, axis='x',which='both',ls='--',color='gray',alpha=0.8)
axsg.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axsg.set_xlim(-0.1,1.1)
axsg.set_ylim(ylimplot)
axsg.set_yticks([])
axsg.patch.set_alpha(0)
axsg.xaxis.set_ticks_position('top')
axsg.spines['left'].set_visible(False)
axsg.spines['right'].set_visible(False)
axsg.set_title('$\sigma_{YSE}$ [GPa]')
axsg.tick_params(labelsize=8)

############################
# Effective friction angle #
############################

axsf = axs.inset_axes((0.800,
                       0,
                       0.10,
                       0.30))
axsf.patch.set_alpha(0.4)

# xdata = np.array([0, 0.05, 1.05, 1.1])
xdata = np.array([0, 0.25, 0.75, 1.0])
ydata = np.array([15, 15, 2, 2])

fisize = 10
axsf.plot(xdata, ydata, 'k-')
axsf.set_xlim([0.10, 0.9])
axsf.set_ylim([0, 17])
axsf.set_xticks([0.25, 0.5, 0.75])
axsf.set_xticklabels([0.05, ' ', 1.05])
axsf.tick_params('x', top=True, labeltop=False)
axsf.xaxis.set_ticks_position('bottom')
axsf.set_xlabel(r"$\varepsilon$", fontsize=fisize)

axsf.set_yticks([2, 15])
axsf.set_yticklabels(['2°', '15°'])
axsf.set_ylabel('$\Phi_{\mathrm{eff}}$', fontsize=fisize)
axsf.tick_params(labelsize=fisize)

axsfC = axsf.twinx()
axsfC.set_ylim([0, 17])
axsfC.set_yticks([2, 15])
axsfC.set_yticklabels([4, 20])
axsfC.set_ylabel('Cohesion [MPa]', fontsize=fisize, rotation=90)
axsfC.tick_params(labelsize=fisize)

#plot velocity bc
if(velocity_from_ascii == True):
    #Velocity = Right side
    vr_plot = np.round(max(abs(VX[:, -1]* (100.0*365.0 * 24.0 * 3600.0))),0)
    fac = 0.96
    axr = axs.inset_axes((fac,
                         0,
                         (1-fac)*2,
                         1))

    scale_veloc = 100.0*365.0*24.0*3600.0
    crt=0

    #Right side velocity arrows
    axr.fill_betweenx((-z + thickness_air) / 1.0e3, VX[:, -1]* (100.0*365.0 * 24.0 * 3600.0), 0, color=None, facecolor=None, hatch='---',alpha=0)
    axr.set_ylim(ylimplot)
    axr.set_yticks([])
    axr.set_xticks([])
    axr.patch.set_alpha(0)
    axr.set_title(f'v = {velocity} [cm/y]', fontsize=8)
    # axr.xaxis.set_ticks_position('top')
    axr.tick_params(labelsize=8)
    axr.set_xlim([-vr_plot, vr_plot])
    axr.spines['left'].set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.spines['bottom'].set_visible(False)

    # Left side velocity arrows
    vl_plot = np.round(max(abs(VX[:, 0]* (100*365.0 * 24.0 * 3600.0))), 0)
    axl = axs.inset_axes((0,
                          0,
                          (1-fac)*2,
                          1))
    
    axl.fill_betweenx((-z + thickness_air) / 1.0e3, VX[:, 0]* (100*365 * 24 * 3600), 0, color=None, facecolor=None, hatch='---',alpha=0)

    axl.set_ylim(-Lz/1000+thickness_air/1000, 0+thickness_air/1000)
    axl.set_yticks([])
    axl.set_xticks([])
    axl.patch.set_alpha(0)
    # axl.set_title(f'v = {v} cm/y')
    # axl.xaxis.set_ticks_position('top')
    # axl.set_xlim([0, 2*velocity])
    axl.set_xlim([0, vl_plot])
    axl.tick_params(labelsize=8)
    axl.spines['left'].set_visible(False)
    axl.spines['right'].set_visible(False)

figname = 'numerical_setup'
fig.savefig(f"{figname}.png", bbox_inches="tight", dpi=300)
fig.savefig(f"{figname}.svg", bbox_inches="tight", dpi=300)
plt.close()

##############################################################################
#Creating run_scripts
##############################################################################

linux = False
mac = False #True
aguia = True
hypatia = True#False

mandyoc_options = '-seed 0,2 -strain_seed 0.0,1.0'

if(linux):
    ncores=12
    run_linux = f'''
            #!/bin/bash
            MPI_PATH=$HOME/opt/petsc/arch-label-optimized/bin
            MANDYOC_PATH=$HOME/opt/mandyoc
            NUMBER_OF_CORES=20
            touch FD.out
            $MPI_PATH/mpirun -{ncores} $NUMBER_OF_CORES $MANDYOC_PATH/mandyoc {mandyoc_options} | tee FD.out
        '''
    with open('run-linux.sh', 'w') as f:
        for line in run_linux.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(mac):
    ncores = 12
    dirname = '${PWD##*/}'
    current_dir = '${PWD}'
    # main_folders = '/scratch/jpmacedo'
    main_folders = '/Users/joao_macedo'
    run_mac = f'''

    #Setup of Mandyoc variables:
    PETSC_DIR='{main_folders}/opt/petsc'
    PETSC_ARCH='optimized-v3.24.1-mpich'

    MANDYOC='{main_folders}/opt/mandyoc/bin/mandyoc'
    MANDYOC_OPTIONS='{mandyoc_options}'

    #run mandyoc
    ${{PETSC_DIR}}/${{PETSC_ARCH}}/bin/mpirun -n {ncores} ${{MANDYOC}} ${{MANDYOC_OPTIONS}}

    conda activate mpy
    #Creating directories for the output files
    bash {main_folders}/opt/mv-updated.sh

    #Creating netdf files
    julia -t {str(int(ncores))} {main_folders}/opt/convertNETCDF_v2.jl {current_dir}
    julia -t {str(int(ncores))} {main_folders}/opt/LithoNETCDF_v2.jl {current_dir}

    # python {main_folders}/opt/track_particles_v3.py {current_dir} 0
    zip {dirname}.zip *.nc

    #run of auxiliary scripts to zip and clean the folder
    bash zipper.sh
    # bash clean.sh
    '''

    with open('run_mac.sh', 'w') as f:
        for line in run_mac.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(aguia):
    # ncores = 150
    aguia = 'aguia4'
    # aguia = 'aguia3'

    if(aguia == 'aguia4'):

        ncores = 160#90
        cores_per_node = 20

        #Estimating number of nodes needed according to number of cores
        nodes = (ncores + cores_per_node - 1) // cores_per_node #ceil division

        partition = 'SP2'
        main_folders = '/temporario2/8672526'

    if(aguia == 'aguia3'):
        partition = 'SP3'
        main_folders =  '/scratch/8672526'

    current_dir = '${PWD}'
    dirname = '${PWD##*/}'
    run_aguia = f'''
            #!/usr/bin/bash
            module load Miniconda

            #SBATCH --partition={partition}
            #SBATCH --ntasks={str(int(ncores))}
            #SBATCH --nodes={nodes}
            #SBATCH --cpus-per-task=1
            #SBATCH --time 192:00:00 #16horas/"2-" para 2 dias com max 8 dias
            #SBATCH --job-name {scenario_name}
            #SBATCH --output slurm_%j.log #ou FD.out/ %j pega o id do job
            #SBATCH --mail-type=BEGIN,FAIL,END
            #SBATCH --mail-user=joao.macedo.silva@usp.br

            export PETSC_DIR='{main_folders}/opt/petsc'
            export PETSC_ARCH='arch-label-optimized'
            MANDYOC='{main_folders}/opt/mandyoc/bin/mandyoc'
            MANDYOC_OPTIONS='{mandyoc_options}'

            $PETSC_DIR/$PETSC_ARCH/bin/mpiexec -n {str(int(ncores))} $MANDYOC $MANDYOC_OPTIONS
            bash zipper.sh
            bash /temporario2/8672526/opt/mv-updated.sh
            #Creating netdf files
            julia -t {str(int(ncores))} /temporario2/8672526/opt/convertNETCDF_v2.jl {current_dir}
            julia -t {str(int(ncores))} /temporario2/8672526/opt/LithoNETCDF_v2.jl {current_dir}

            python /temporario2/8672526/opt/frames_generator.py
            
            zip {dirname}.zip *.nc
            bash clean.sh
        
        '''
    
    with open('run_aguia.sh', 'w') as f:
        for line in run_aguia.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(hypatia):
    ncores=96
    # ncores = 192
    # ncores = 256
    # ncores = 384
    cores_per_node = 96

    #Estimating number of nodes needed according to number of cores
    nodes = (ncores + cores_per_node - 1) // cores_per_node #ceil division

    dirname = '${PWD##*/}'
    current_dir = '${PWD}'
    # main_folders = '/scratch/jpmacedo'
    main_folders = '/home/jpmacedo'
    run_hypatia = f'''
    #!/usr/bin/env bash
    #SBATCH --mail-type=BEGIN,END,FAIL         			# Mail events (NONE, BEGIN, END, FAIL, ALL)
    #SBATCH --mail-user=joao.macedo.silva@usp.br		# Where to send mail
    #SBATCH --ntasks={str(int(ncores))}
    #SBATCH --nodes={str(int(nodes))}
    #SBATCH --cpus-per-task=1
    #SBATCH --hint=nomultithread
    #SBATCH --exclude=f001
    #SBATCH --time 72:00:00 # 16 horas; poderia ser “2-” para 2 dias; máximo “8-”
    #SBATCH --job-name {scenario_name}
    #SBATCH --output slurm_{scenario_name}_%j.log
    #SBATCH --error=log_error_{scenario_name}_%j.log
    #SBATCH --no-requeue

    module purge
    module load gcc/13.2.0-gcc-8.5.0-tnbqzki
    module load openmpi/5.0.3-gcc-8.5.0-no4tqjk
    module load cmake/3.27.9-gcc-8.5.0-33534nt

    #Setup of Mandyoc variables:
    PETSC_DIR='{main_folders}/opt/petsc'
    PETSC_ARCH='optimized-v3.24.1-openmpi'

    MANDYOC='{main_folders}/opt/mandyoc/bin/mandyoc'
    MANDYOC_OPTIONS='{mandyoc_options}'

    #run mandyoc
    mpirun -n ${{SLURM_NTASKS}} --map-by :OVERSUBSCRIBE ${{MANDYOC}} ${{MANDYOC_OPTIONS}}

    # # conda activate mpy
    # #Creating directories for the output files
    # bash zipper.sh
    # bash /home/jpmacedo/opt/mv-updated.sh
    
    # #Creating netdf files
    # julia -t {str(int(ncores))} /home/jpmacedo/opt/convertNETCDF_v2.jl {current_dir}
    # julia -t {str(int(ncores))} /home/jpmacedo/opt/LithoNETCDF_v2.jl {current_dir}

    zip {dirname}.zip *.hdf5

    #run of auxiliary scripts to zip and clean the folder
    
    # bash clean.sh
    '''
    with open('run_hypatia.sh', 'w') as f:
        for line in run_hypatia.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')


# zipper = f'''
#         #!/usr/bin/env bash
#         DIRNAME={dirname}

#         # Primeiro zipa os arquivos fixos
#         zip "$DIRNAME.zip" interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh

#         # Lista de padrões
#         patterns=(
#             "bc_velocity_*.txt"
#             "density_*.txt"
#             "heat_*.txt"
#             "pressure_*.txt"
#             "surface*.txt"
#             "litho*.txt"
#             "strain_*.txt"
#             "temperature_*.txt"
#             "time_*.txt"
#             "velocity_*.txt"
#             "viscosity_*.txt"
#             "scale_bcv.txt"
#             "step*.txt"
#             "Phi*.txt"
#             "dPhi*.txt"
#             "X_depletion*.txt"
#             "*.bin*.txt"
#             "bc*-1.txt"
#             "*.log"
#             # "_*.nc"
#             )

#         # Faz um loop e usa find para evitar o erro "argument list too long"
#         for pat in "${{patterns[@]}}"; do
#             find . -maxdepth 1 -type f -name "$pat" -exec zip -u -r "$DIRNAME.zip" {{}} +
#         done
#     '''
zipper = f'''
        #!/usr/bin/env bash
        DIRNAME={dirname}

        # Primeiro zipa os arquivos fixos
        zip "$DIRNAME.zip" interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh

        # Lista de padrões
        patterns=(
            "bc_velocity_*.txt"
            "density"
            "heat"
            "pressure"
            "surface"
            "lithos"
            "strain"
            "strain_rate"
            "temperature"
            "time"
            "velocity"
            "viscosity"
            "scale_bcv.txt"
            "steps"
            "Phi"
            "dPhi"
            "X_depletion"
            "*.bin*.txt"
            "bc*-1.txt"
            "*.log"
            # "_*.nc"
            )

        # Faz um loop e usa find para evitar o erro "argument list too long"
        for pat in "${{patterns[@]}}"; do
            find . -maxdepth 1 -type f -name "$pat" -exec zip -u -r "$DIRNAME.zip" {{}} +
        done
    '''
with open('zipper.sh', 'w') as f:
    for line in zipper.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

clean = f'''
        #!/usr/bin/env bash

        # Lista de padrões
        patterns=(
            "bc_velocity_*.txt"
            "density_*.txt"
            "heat_*.txt"
            "pressure_*.txt"
            "surface*.txt"
            "litho*.txt"
            "strain_*.txt"
            "temperature_*.txt"
            "time_*.txt"
            "velocity_*.txt"
            "viscosity_*.txt"
            "scale_bcv.txt"
            "step*.txt"
            "Phi*.txt"
            "dPhi*.txt"
            "X_depletion*.txt"
            "*.bin*.txt"
            "bc*-1.txt"
            )

        # Para cada padrão, procurar e remover com segurança
        for pat in "${{patterns[@]}}"; do
            find . -maxdepth 1 -type f -name "$pat" -exec rm -f {{}} +
        done
    '''
with open('clean.sh', 'w') as f:
    for line in clean.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

#zip input files
filename = 'inputs_'+path[-1]+'.zip'
files_list = ' infos*.txt interfaces.txt param.txt input*_0.txt run*.sh vel*.txt scale_bcv.txt *.png precipitation.txt climate.txt zipper.sh clean.sh'
os.system('zip '+filename+files_list)



##############################################################################
# Scenario infos
##############################################################################

print(f"Scenario kind: {experiemnts[scenario_kind]}")
print(f"N cores: {ncores}")
print('Domain parameters:')
print(f"\tLx: {Lx*1.0e-3} km")
print(f"\tLz: {Lz*1.0e-3} km")
print(f"\tNx: {Nx}")
print(f"\tNz: {Nz}")
print(f"Resolution dx x dz: {1.0e-3*Lx/(Nx-1)} x {1.0e-3*Lz/(Nz-1)} km2")
print(f'Time limit: {time_max/1.0e6} Myr')
if(variable_bcv == True):
    print(f'Time of rifting1: {dt_rifting1} Myr')
    print(f"Time of quiescence after rifting: {dt_quiescence1} Myr")
    print(f"Time of quiescence after orogeny: {dt_quiescence2} Myr")
print(f'Total time: {time_max/1.0e6} Myr')
print('Layers thickness:')
print(f"\tair: {thickness_air*1.0e-3} km")
if(sediments==True):
    print(f"\tsediments: {thickness_sed/1000} km")
    print(f"\tdecolement: {thickness_decolement/1000} km")
print(f"\tupper crust: {thickness_upper_crust*1.0e-3} km")
print(f"\tlower crust: {thickness_lower_crust*1.0e-3} km")
print(f"\tnon cratonic mantle lithosphere: {thickness_litho/1000} km")
if(sediments==True):
    print(f"\tcrust: {(thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust)/1000} km")
else:
    print(f"\tcrust: {(thickness_upper_crust + thickness_lower_crust)/1000} km")
print(f"\tnon cratonic lithosphere: {thickness_litho*1.0e-3} km")
print('Important scale factors (C):')
print(f"\tair: {air.effective_viscosity_scale_factor}")
if(sediments==True):
    print(f"\tsediments: {sediments.effective_viscosity_scale_factor}")
    print(f"\tdecolement: {decolement.effective_viscosity_scale_factor}")
print(f"\tupper crust: {upper_crust.effective_viscosity_scale_factor}")
print(f"\tlower crust: {lower_crust.effective_viscosity_scale_factor}")
print(f"\tweak seed: {seed_base.effective_viscosity_scale_factor}")
print(f"\tnon cratonic mantle lithosphere: {lithospheric_mantle.effective_viscosity_scale_factor}")
print(f"Preset of initial temperature field: {preset}")
print(f"Radiogenic heat in lithospheric mantle: {lithospheric_mantle.radiogenic_heat_production}")
print(f"Surface process: {sp_surface_processes}")
print(f"Velocity field: {velocity_from_ascii}")
print(f"Variable velocity field: {variable_bcv}")
print(f"Climate change: {climate_change_from_ascii}")
print(f"Periodic Boundary: {periodic_boundary}")
print('Initial temperature field setup:')
print(f"\tPreset of initial temperature field: {preset}")
print(f"\tIncrease in mantle basal temperature (Ta): {DeltaT} oC")
print(f"\tAssumed mantle Potential Temperature for diffusive model: {TP} oC")
print(f'magmatism: {magmatism}')
print(f'rheology model in param file: {rheology_model}')

#Save scenario infos
scenario_infos = ['SCENARIO INFOS:']
scenario_infos.append(' ')
scenario_infos.append('Name: ' + path[-1])
scenario_infos.append(f"Scenario kind: {experiemnts[scenario_kind]}")
scenario_infos.append(f"N cores: {ncores}")
scenario_infos.append(' ')
scenario_infos.append('Domain parameters:')
scenario_infos.append(f"\tLx: {Lx*1.0e-3} km")
scenario_infos.append(f"\tLz: {Lz*1.0e-3} km")
scenario_infos.append(f"\tNx: {Nx}")
scenario_infos.append(f"\tNz: {Nz}")
scenario_infos.append(f"Resolution dx x dz: {1.0e-3*Lx/(Nx-1)} x {1.0e-3*Lz/(Nz-1)} km2")
scenario_infos.append(' ')
scenario_infos.append(f'Time limit: {time_max/1.0e6} Myr')
if(variable_bcv == True):
    scenario_infos.append(f'Time of rifting1: {dt_rifting1} Myr')
    scenario_infos.append(f"Time of quiescence after rifting: {dt_quiescence1} Myr")
    scenario_infos.append(f"Time of quiescence after orogeny: {dt_quiescence2} Myr")
    scenario_infos.append(' ')
scenario_infos.append(f'Total time: {time_max/1.0e6} Myr')
scenario_infos.append('Layers thickness:')
scenario_infos.append(f"\tair: {thickness_air*1.0e-3} km")
if(sediments==True):
    scenario_infos.append(f"\tsediments: {thickness_sed/1000} km")
    scenario_infos.append(f"\tdecolement: {thickness_decolement/1000} km")
scenario_infos.append(f"\tupper crust: {thickness_upper_crust*1.0e-3} km")
scenario_infos.append(f"\tlower crust: {thickness_lower_crust*1.0e-3} km")
scenario_infos.append(f"\tnon cratonic mantle lithosphere:{ thickness_litho} km")
if(sediments==True):
    scenario_infos.append(f"\tcrust: {(thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust)/1000}")
else:
    scenario_infos.append(f"\tcrust: {(thickness_upper_crust + thickness_lower_crust)/1000}")
scenario_infos.append(f"\tnon cratonic lithosphere: {thickness_litho*1.0e-3} km")
scenario_infos.append(' ')
scenario_infos.append(' ')
scenario_infos.append('Important scale factors (C):')
scenario_infos.append(f"\tair: {air.effective_viscosity_scale_factor}")
if(sediments==True):
    scenario_infos.append(f"\tsediments: {sediments.effective_viscosity_scale_factor}")
    scenario_infos.append(f"\tdecolement: {decolement.effective_viscosity_scale_factor}")
scenario_infos.append(f"\tupper crust: {upper_crust.effective_viscosity_scale_factor}")
scenario_infos.append(f"\tlower crust: {lower_crust.effective_viscosity_scale_factor}")
scenario_infos.append(f"\tweak seed: {seed_base.effective_viscosity_scale_factor}")
scenario_infos.append(f"\tnon cratonic mantle lithosphere: {lithospheric_mantle.effective_viscosity_scale_factor}")
scenario_infos.append(' ')
scenario_infos.append(f"Preset of initial temperature field: {preset}")
scenario_infos.append(f"Radiogenic heat in lithospheric mantle: {lithospheric_mantle.radiogenic_heat_production}")
scenario_infos.append(f"Surface process: {sp_surface_processes}")
scenario_infos.append(f"Velocity field: {velocity_from_ascii}")
if(velocity_from_ascii==True):
    scenario_infos.append(f"inital velocity: {vL*(365 * 24 * 3600)*2*100} cm/yr") 
scenario_infos.append(f"Variable velocity field: {variable_bcv}")
scenario_infos.append(f"Climate change: {climate_change_from_ascii}")
scenario_infos.append(f"Periodic Boundary: {periodic_boundary}")
scenario_infos.append('Initial temperature field setup:')
scenario_infos.append(f"\tPreset of initial temperature field: {preset}")
scenario_infos.append(f"\tIncrease in mantle basal temperature (Ta): {DeltaT} oC")
scenario_infos.append(f"\tAssumed mantle Potential Temperature for diffusive model: {TP} oC")
scenario_infos.append(' ')
scenario_infos.append(f'magmatism: {magmatism}')
scenario_infos.append(f'rheology model in param file: {rheology_model}')

np.savetxt('infos_'+path[-1] + '.txt', scenario_infos, fmt="%s")
