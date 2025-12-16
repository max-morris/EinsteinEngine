from enum import Enum


class EvolutionThorn(Enum):
    CANUDAX = 0,
    COTTONMOUTH = 1,
    Z4C = 2,


USED_EVOLUTION_THORN = EvolutionThorn.COTTONMOUTH

if USED_EVOLUTION_THORN == EvolutionThorn.CANUDAX:
    evolution_thorn_name = "CanudaX_BSSNMoL"
    evolution_thorn_settings = r"""
CanudaX_BSSNMoL::calculate_constraints              = yes
CanudaX_BSSNMoL::calculate_constraints_every        = 256
CanudaX_BSSNMoL::slicing_condition                  = "1+log"
CanudaX_BSSNMoL::kappa_alpha                        = 2.0
CanudaX_BSSNMoL::impose_conf_fac_floor_at_initial   = yes
CanudaX_BSSNMoL::conf_fac_floor                     = 1.0d-04
CanudaX_BSSNMoL::h_amp                              = 1.0
CanudaX_BSSNMoL::sigma                              = 30
CanudaX_BSSNMoL::precollapsed_lapse                 = yes
CanudaX_BSSNMoL::zeta_alpha                         = 1
CanudaX_BSSNMoL::zeta_beta                          = 1
CanudaX_BSSNMoL::eta_beta                           = 1
CanudaX_BSSNMoL::beta_Gamma                         = 0.75
CanudaX_BSSNMoL::chi_gamma                          = 0.667
CanudaX_BSSNMoL::beta_Alp                           = 1
CanudaX_BSSNMoL::eta_transition                     = no
CanudaX_BSSNMoL::moving_eta_transition              = no
CanudaX_BSSNMoL::eta_beta_dynamic                   = no
CanudaX_BSSNMoL::eps_r                              = 1e-5
CanudaX_BSSNMoL::reset_dethh                        = yes
CanudaX_BSSNMoL::make_aa_tracefree                  = yes
CanudaX_BSSNMoL::stress_energy_state                = yes
CanudaX_BSSNMoL::derivs_order                       = 4
CanudaX_BSSNMoL::use_advection_stencils             = yes
CanudaX_BSSNMoL::add_KO_dissipation                 = yes
CanudaX_BSSNMoL::diss_eps                           = 0.15
CanudaX_BSSNMoL::diss_order                         = 5
CanudaX_BSSNMoL::use_local_error_estimate           = no
CanudaX_BSSNMoL::boundary_conditions                = "radiative"
"""
    evolution_thorn_hc_var = "CanudaX_BSSNMoL::ham"
    evolution_thorn_mc_var = "CanudaX_BSSNMoL::mom"

elif USED_EVOLUTION_THORN == EvolutionThorn.COTTONMOUTH:
    evolution_thorn_name = "CottonmouthBSSNOK"
    evolution_thorn_settings = r"""
CottonmouthBSSNOK::eta_b                  = 1.0
CottonmouthBSSNOK::conformal_factor_floor = 1.0e-6
CottonmouthBSSNOK::evolved_lapse_floor    = 1.0e-8
CottonmouthBSSNOK::dissipation_epsilon    = 0.32
"""
    evolution_thorn_hc_var = "CottonmouthBSSNOK::HamCons"
    evolution_thorn_mc_var = "CottonmouthBSSNOK::MomCons"

elif USED_EVOLUTION_THORN == EvolutionThorn.Z4C:
    evolution_thorn_name = "Z4c"
    evolution_thorn_settings = r"""
Z4c::calc_ADM_vars = yes
Z4c::calc_ADMRHS_vars = no
Z4c::calc_constraints = yes
Z4c::chi_floor = 1.0e-6
Z4c::alphaG_floor = 1.0e-8
Z4c::epsdiss = 0.32
Z4c::boundary_conditions = "NewRadX"
"""
    evolution_thorn_name = "Z4c::HC"
    evolution_thorn_settings = "Z4c::MtC"

template = rf"""
###############################
# Simple test of magnetised TOV neutron star
# Same neutron star as the gallery example
# K=100, rho_c = 1.28e-3 => M = 1.4, M_b = 1.506
# evolve for t = 48.82 M
# This file is adapted from AsterX/test/magTOV_Z4c_AMR.par
##############################

ActiveThorns = "
    ADMBaseX
    BoxInBox
    CarpetX
    HydroBaseX
    IOUtil
    ODESolvers
    TmunuBaseX
    AsterX
    AsterSeeds
    {evolution_thorn_name}
    AsterMasks
    TOVSolverX
    SystemTopology
"

# Termination
Cactus::terminate       = "any"
Cactus::cctk_itlast     = 10000000
Cactus::cctk_final_time = 400
Cactus::max_runtime     = 1 * 24 * 60 - 30

# General settings
CarpetX::verbose    = yes
ODESolvers::verbose = yes

Cactus::presync_mode = "mixed-error"
CarpetX::poison_undefined_values = no

# Distribution
CarpetX::blocking_factor_x = 8
CarpetX::blocking_factor_y = 8
CarpetX::blocking_factor_z = 8

# Grid settings
# Finest grid spacing is 0.2509765625
CarpetX::xmin = -642.5
CarpetX::ymin = -642.5
CarpetX::zmin = -642.5

CarpetX::xmax = 642.5
CarpetX::ymax = 642.5
CarpetX::zmax = 642.5

CarpetX::ncells_x = 160
CarpetX::ncells_y = 160
CarpetX::ncells_z = 160

CarpetX::boundary_x       =  "neumann"
CarpetX::boundary_y       =  "neumann"
CarpetX::boundary_z       =  "neumann"
CarpetX::boundary_upper_x =  "neumann"
CarpetX::boundary_upper_y =  "neumann"
CarpetX::boundary_upper_z =  "neumann"

CarpetX::ghost_size = 3

# Mesh refinement
CarpetX::max_num_levels         = 6
CarpetX::regrid_every           = 128
CarpetX::regrid_error_threshold = 0.9

CarpetX::prolongation_type = "ddf"

BoxInBox::num_regions  = 1
BoxInBox::shape_1      = "cube"
BoxInBox::num_levels_1 = 6
BoxInBox::radius_1     = [ -1.0, 321.25, 160.625, 80.3125, 40.15625, 20.078125 ]

# Time integration
ODESolvers::method = "RK4"
CarpetX::dtfac     = 0.25

# Spacetime evolution
ADMBaseX::initial_data                       = "tov"
ADMBaseX::initial_lapse                      = "tov"
ADMBaseX::initial_shift                      = "tov"
ADMBaseX::initial_dtlapse                    = "zero"
ADMBaseX::initial_dtshift                    = "zero"
{evolution_thorn_settings}
# Hydro
TOVSolverX::TOV_Rho_Central[0] = 1.28e-3
TOVSolverX::TOV_Gamma          = 2.0
TOVSolverX::TOV_K              = 100.0
TOVSolverX::TOV_Cowling = no

AsterSeeds::test_type = "3DTest"
AsterSeeds::test_case = "magTOV"
AsterSeeds::Afield_config = "internal dipole"
AsterSeeds::Ab = 100.0
AsterSeeds::press_cut = 0.04
AsterSeeds::press_max = 1.638e-4
AsterSeeds::Avec_kappa = 2.0

AsterX::debug_mode = "no"
AsterX::flux_type = "HLLE"
AsterX::vector_potential_gauge = "algebraic"
AsterX::local_spatial_order = 4
AsterX::local_estimate_error = "no"

ReconX::reconstruction_method = "PPM"
ReconX::ppm_zone_flattening = "yes"
ReconX::ppm_shock_detection = "no"

Con2PrimFactory::c2p_prime = "Noble"
Con2PrimFactory::c2p_second = "Palenzuela"
Con2PrimFactory::c2p_tol = 1e-8
Con2PrimFactory::max_iter = 100
Con2PrimFactory::rho_abs_min = 1e-11
Con2PrimFactory::atmo_tol = 1e-3
Con2PrimFactory::unit_test = "yes"
Con2PrimFactory::B_lim = 1e8
Con2PrimFactory::vw_lim = 1e8
Con2PrimFactory::Ye_lenient = "yes"

EOSX::evolution_eos = "IdealGas"
EOSX::gl_gamma = 2.0
EOSX::poly_gamma = 2.0
EOSX::poly_k = 100
EOSX::rho_max = 1e8
EOSX::eps_max = 1e8

# IO
IO::out_dir = $parfile
IO::out_every = 64

CarpetX::out_metadata    = no
CarpetX::out_performance = no

CarpetX::out_norm_every         = 32
CarpetX::out_norm_omit_unstable = no
CarpetX::out_norm_vars          = "
  HydroBaseX::rho
  {evolution_thorn_hc_var}
  {evolution_thorn_mc_var}
"

CarpetX::out_silo_vars = "
    HydroBaseX::Bvec
    HydroBaseX::rho
    HydroBaseX::vel
    HydroBaseX::eps
    HydroBaseX::press
    ADMBaseX::lapse
    ADMBaseX::shift
    ADMBaseX::metric
    TmunuBaseX::eTtt
    TmunuBaseX::eTti
    TmunuBaseX::eTij
    {evolution_thorn_hc_var}
    {evolution_thorn_mc_var}
"

CarpetX::out_tsv = no
CarpetX::out_tsv_vars = ""

# Checkpoint and recovery
IO::checkpoint_dir                  = "../checkpoints_$parfile"
IO::recover_dir                     = "../checkpoints_$parfile"
IO::checkpoint_ID                   = no
IO::checkpoint_every_walltime_hours = 6
IO::checkpoint_on_terminate         = yes
IO::recover                         = "autoprobe"
CarpetX::checkpoint_method          = "openpmd"
CarpetX::recover_method             = "openpmd"
"""

with open("mag_TOV.par", "w") as file:
    file.write(template)
