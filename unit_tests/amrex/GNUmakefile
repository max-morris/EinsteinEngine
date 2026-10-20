# The AMReX backend's gate (docs, section 9, items 6 and 7): builds main.cpp with the generated toy headers in
# GEN_DIR, for CPU or CUDA (AMReX GNU make options on the command line). Needs AMREX_HOME.
AMREX_HOME ?= $(error set AMREX_HOME)
GEN_DIR ?= gen
DEBUG = FALSE
USE_MPI = FALSE
USE_OMP = FALSE
DIM = 3
COMP ?= gnu
CXXSTD = c++20
EBASE = AMReXGate
BL_NO_FORT = TRUE
include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include $(AMREX_HOME)/Src/Base/Make.package
CEXE_sources += main.cpp
VPATH_LOCATIONS += $(realpath .)
INCLUDE_LOCATIONS += $(realpath .) $(realpath $(GEN_DIR))
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
