# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation

# Include any dependencies generated for this target.
include CMakeFiles/vector_mean_curvature.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/vector_mean_curvature.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vector_mean_curvature.dir/flags.make

CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o: CMakeFiles/vector_mean_curvature.dir/flags.make
CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o: vector_mean_curvature.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o"
	/Applications/deal.II.app/Contents/Resources/opt/openmpi-1.10.2/bin/mpic++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o -c /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation/vector_mean_curvature.cc

CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.i"
	/Applications/deal.II.app/Contents/Resources/opt/openmpi-1.10.2/bin/mpic++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation/vector_mean_curvature.cc > CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.i

CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.s"
	/Applications/deal.II.app/Contents/Resources/opt/openmpi-1.10.2/bin/mpic++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation/vector_mean_curvature.cc -o CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.s

CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o.requires:

.PHONY : CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o.requires

CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o.provides: CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o.requires
	$(MAKE) -f CMakeFiles/vector_mean_curvature.dir/build.make CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o.provides.build
.PHONY : CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o.provides

CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o.provides.build: CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o


# Object files for target vector_mean_curvature
vector_mean_curvature_OBJECTS = \
"CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o"

# External object files for target vector_mean_curvature
vector_mean_curvature_EXTERNAL_OBJECTS =

vector_mean_curvature: CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o
vector_mean_curvature: CMakeFiles/vector_mean_curvature.dir/build.make
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/lib/libdeal_II.g.8.4.1.dylib
vector_mean_curvature: /usr/lib/libbz2.dylib
vector_mean_curvature: /usr/lib/libz.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/petsc-3e25e16/lib/libparmetis.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/petsc-3e25e16/lib/libmetis.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtrilinoscouplings.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libpiro.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/librol.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstokhos_muelu.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstokhos_ifpack2.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstokhos_amesos2.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstokhos_tpetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstokhos_sacado.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstokhos.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/librythmos.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libmuelu-adapters.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libmuelu-interface.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libmuelu.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/liblocathyra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/liblocaepetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/liblocalapack.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libloca.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libnoxepetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libnoxlapack.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libnox.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libintrepid.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libteko.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstratimikos.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstratimikosbelos.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstratimikosaztecoo.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstratimikosamesos.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstratimikosml.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libstratimikosifpack.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libifpack2-adapters.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libifpack2.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libanasazitpetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libModeLaplace.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libanasaziepetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libanasazi.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libkomplex.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libamesos2.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libbelostpetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libbelosepetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libbelos.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libml.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libifpack.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libzoltan2.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libpamgen_extras.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libpamgen.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libamesos.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libgaleri-xpetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libgaleri-epetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libaztecoo.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libisorropia.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/liboptipack.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libxpetra-sup.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libxpetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libthyratpetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libthyraepetraext.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libthyraepetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libthyracore.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libepetraext.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtpetraext.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtpetrainout.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtpetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libkokkostsqr.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtpetrakernels.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtpetraclassiclinalg.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtpetraclassicnodeapi.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtpetraclassic.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libtriutils.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libglobipack.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libshards.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libzoltan.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libepetra.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libsacado.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/librtop.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libteuchoskokkoscomm.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libteuchoskokkoscompat.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libteuchosremainder.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libteuchosnumerics.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libteuchoscomm.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libteuchosparameterlist.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libteuchoscore.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libkokkosalgorithms.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libkokkoscontainers.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/trilinos-0b08cd5/lib/libkokkoscore.dylib
vector_mean_curvature: /usr/lib/liblapack.dylib
vector_mean_curvature: /usr/lib/libblas.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/openmpi-1.10.2/lib/libmpi_cxx.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/arpack-ng-d66b8b4/lib/libparpack.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/arpack-ng-d66b8b4/lib/libarpack.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/petsc-3e25e16/lib/libhdf5_hl.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/petsc-3e25e16/lib/libhdf5.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKBO.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKBool.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKBRep.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKernel.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKFeat.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKFillet.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKG2d.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKG3d.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKGeomAlgo.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKGeomBase.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKHLR.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKIGES.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKMath.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKMesh.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKOffset.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKPrim.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKShHealing.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKSTEP.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKSTEPAttr.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKSTEPBase.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKSTL.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKTopAlgo.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/oce-c0cdb53/lib/libTKXSBase.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/p4est-8d811a8/lib/libp4est.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/p4est-8d811a8/lib/libsc.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/openmpi-1.10.2/lib/libmpi.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/slepc-2c065dd/lib/libslepc.dylib
vector_mean_curvature: /Applications/deal.II.app/Contents/Resources/opt/petsc-3e25e16/lib/libpetsc.dylib
vector_mean_curvature: CMakeFiles/vector_mean_curvature.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable vector_mean_curvature"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_mean_curvature.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vector_mean_curvature.dir/build: vector_mean_curvature

.PHONY : CMakeFiles/vector_mean_curvature.dir/build

CMakeFiles/vector_mean_curvature.dir/requires: CMakeFiles/vector_mean_curvature.dir/vector_mean_curvature.cc.o.requires

.PHONY : CMakeFiles/vector_mean_curvature.dir/requires

CMakeFiles/vector_mean_curvature.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vector_mean_curvature.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vector_mean_curvature.dir/clean

CMakeFiles/vector_mean_curvature.dir/depend:
	cd /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation /Users/stephenstd/gpde/code/dealii/sandbox/vector_mean_curvature_computation/CMakeFiles/vector_mean_curvature.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vector_mean_curvature.dir/depend
