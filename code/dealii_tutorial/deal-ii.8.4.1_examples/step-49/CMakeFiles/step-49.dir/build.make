# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49

# Include any dependencies generated for this target.
include CMakeFiles/step-49.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/step-49.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/step-49.dir/flags.make

CMakeFiles/step-49.dir/step-49.cc.o: CMakeFiles/step-49.dir/flags.make
CMakeFiles/step-49.dir/step-49.cc.o: step-49.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/step-49.dir/step-49.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/step-49.dir/step-49.cc.o -c /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49/step-49.cc

CMakeFiles/step-49.dir/step-49.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/step-49.dir/step-49.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49/step-49.cc > CMakeFiles/step-49.dir/step-49.cc.i

CMakeFiles/step-49.dir/step-49.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/step-49.dir/step-49.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49/step-49.cc -o CMakeFiles/step-49.dir/step-49.cc.s

CMakeFiles/step-49.dir/step-49.cc.o.requires:
.PHONY : CMakeFiles/step-49.dir/step-49.cc.o.requires

CMakeFiles/step-49.dir/step-49.cc.o.provides: CMakeFiles/step-49.dir/step-49.cc.o.requires
	$(MAKE) -f CMakeFiles/step-49.dir/build.make CMakeFiles/step-49.dir/step-49.cc.o.provides.build
.PHONY : CMakeFiles/step-49.dir/step-49.cc.o.provides

CMakeFiles/step-49.dir/step-49.cc.o.provides.build: CMakeFiles/step-49.dir/step-49.cc.o

# Object files for target step-49
step__49_OBJECTS = \
"CMakeFiles/step-49.dir/step-49.cc.o"

# External object files for target step-49
step__49_EXTERNAL_OBJECTS =

step-49: CMakeFiles/step-49.dir/step-49.cc.o
step-49: CMakeFiles/step-49.dir/build.make
step-49: /usr/local/lib/libdeal_II.g.so.8.4.1
step-49: CMakeFiles/step-49.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable step-49"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/step-49.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/step-49.dir/build: step-49
.PHONY : CMakeFiles/step-49.dir/build

CMakeFiles/step-49.dir/requires: CMakeFiles/step-49.dir/step-49.cc.o.requires
.PHONY : CMakeFiles/step-49.dir/requires

CMakeFiles/step-49.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/step-49.dir/cmake_clean.cmake
.PHONY : CMakeFiles/step-49.dir/clean

CMakeFiles/step-49.dir/depend:
	cd /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49 /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49 /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49 /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49 /home/tds3/gpde/code/dealii_tutorial/deal-ii.8.4.1_examples/step-49/CMakeFiles/step-49.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/step-49.dir/depend

