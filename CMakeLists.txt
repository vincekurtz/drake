# -*- mode: cmake -*-
# vi: set ft=cmake :

# The minimum required CMake version should match the minimum listed in
# doc/_pages/from_source.md, which is also the minimum installed by
# setup/install_prereqs across all platforms.
# When this changes, drake-external-examples/drake_cmake_external/CMakeLists.txt
# should be changed accordingly so that downstream users are synced with
# Drake itself.
cmake_minimum_required(VERSION 3.22)

project(drake
  DESCRIPTION "Model-based design and verification for robotics"
  LANGUAGES C CXX
)

# The primary build system for Drake is Bazel (https://bazel.build/). For CMake,
# our objective is to accept configuration options using their standard spelling
# (e.g., `-DCMAKE_BUILD_TYPE=Release`) and install Drake using those settings.
#
# We'll do that by converting the settings to generated Bazel inputs:
# - a generated `MODULE.bazel` that depends on the Drake module and customizes
#   the toolchain selection.
# - a `.bazelrc` file that specifies configuration choices.
# and then running the `@drake//:install` program from that temporary workspace.

list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake/modules")

include(CTest)

configure_file(CTestCustom.cmake.in CTestCustom.cmake @ONLY)

if(ANDROID OR CYGWIN OR IOS OR NOT UNIX)
  message(FATAL_ERROR
    "Android, Cygwin, iOS, and non-Unix platforms are NOT supported"
  )
endif()

set(BAZELRC_IMPORTS "tools/bazel.rc")
set(UNIX_DISTRIBUTION_CODENAME)

# The minimum Darwin version and codename should correspond to the minimum
# supported macOS version listed in doc/_pages/installation.md and
# doc/_pages/from_source.md.
# For Ubuntu, don't make as harsh a check for minimum supported version
# since Drake *can* compile on non-Ubuntu UNIX, even though it's not
# officially supported.
set(MINIMUM_DARWIN_VERSION 23)
set(MINIMUM_MACOS_CODENAME "Sonoma")

if(APPLE)
  if(CMAKE_SYSTEM_VERSION VERSION_LESS ${MINIMUM_DARWIN_VERSION})
    message(WARNING "Darwin ${CMAKE_SYSTEM_VERSION} is NOT supported. "
      "Please use Darwin ${MINIMUM_DARWIN_VERSION}.x (macOS "
      "${MINIMUM_MACOS_CODENAME}) or newer."
    )
  endif()

  list(APPEND BAZELRC_IMPORTS "tools/macos.bazelrc")

  execute_process(
    COMMAND "/usr/bin/arch"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MACOS_ARCH)
  if(MACOS_ARCH STREQUAL "")
    message(FATAL_ERROR "Could NOT query macOS arch")
  endif()
  list(APPEND BAZELRC_IMPORTS "tools/macos-arch-${MACOS_ARCH}.bazelrc")
else()
  list(APPEND BAZELRC_IMPORTS "tools/ubuntu.bazelrc")

  find_program(LSB_RELEASE_EXECUTABLE NAMES lsb_release)
  mark_as_advanced(LSB_RELEASE_EXECUTABLE)
  if(NOT LSB_RELEASE_EXECUTABLE)
    message(WARNING "Could NOT find the lsb_release executable")
  else()
    execute_process(COMMAND "${LSB_RELEASE_EXECUTABLE}" --codename --short
      RESULT_VARIABLE LSB_RELEASE_CODENAME_SHORT_RESULT_VARIABLE
      OUTPUT_VARIABLE UNIX_DISTRIBUTION_CODENAME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT LSB_RELEASE_CODENAME_SHORT_RESULT_VARIABLE EQUAL 0)
      message(WARNING "Could NOT run the lsb_release executable")
    else()
      set(MAYBE_RC "tools/ubuntu-${UNIX_DISTRIBUTION_CODENAME}.bazelrc")
      if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${MAYBE_RC}")
        message(WARNING "Could not find config file ${MAYBE_RC}. "
          "This may indicate that you're using an OS or OS version that "
          "is not officially supported, so the build configuration cannot be "
          "fine-tuned. See https://drake.mit.edu/from_source.html for details "
          "on officially supported platforms when building from source.")
      else()
        list(APPEND BAZELRC_IMPORTS "${MAYBE_RC}")
      endif()
    endif()
  endif()
endif()

# This version number should match bazel_compatibility in MODULE.bazel.
set(MINIMUM_BAZEL_VERSION 7.4.1)
find_package(Bazel ${MINIMUM_BAZEL_VERSION} MODULE)
if(NOT Bazel_FOUND)
  set(Bazel_EXECUTABLE "${PROJECT_SOURCE_DIR}/third_party/com_github_bazelbuild_bazelisk/bazelisk.py")
  message(STATUS "Using Bazelisk as Bazel_EXECUTABLE to fetch Bazel on demand")
endif()

get_filename_component(C_COMPILER_REALPATH "${CMAKE_C_COMPILER}" REALPATH)
get_filename_component(C_COMPILER_NAME "${C_COMPILER_REALPATH}" NAME)

get_filename_component(CXX_COMPILER_REALPATH "${CMAKE_CXX_COMPILER}" REALPATH)
get_filename_component(CXX_COMPILER_NAME "${CXX_COMPILER_REALPATH}" NAME)

if(C_COMPILER_NAME STREQUAL ccache OR CXX_COMPILER_NAME STREQUAL ccache)
  message(FATAL_ERROR
    "Compilation with ccache is NOT supported due to incompatibility with Bazel"
  )
endif()

# Get the full C++ compiler major version for our cmake/bazel.rc.in.
string(REGEX MATCH "^([0-9]+)"
  DRAKE_CC_TOOLCHAIN_COMPILER_MAJOR
  "${CMAKE_CXX_COMPILER_VERSION}")
if(NOT DRAKE_CC_TOOLCHAIN_COMPILER_MAJOR)
  set(DRAKE_CC_TOOLCHAIN_COMPILER_MAJOR "0")
endif()

# The minimum compiler versions should match those listed in both
# doc/_pages/from_source.md and tools/workspace/cc/repository.bzl.
set(MINIMUM_APPLE_CLANG_VERSION 16)
set(MINIMUM_CLANG_VERSION 15)
set(MINIMUM_GNU_VERSION 11)

if(CMAKE_C_COMPILER_ID STREQUAL AppleClang)
  if(CMAKE_C_COMPILER_VERSION VERSION_LESS ${MINIMUM_APPLE_CLANG_VERSION})
    message(WARNING
      "Compilation with clang ${CMAKE_C_COMPILER_VERSION} is NOT supported"
    )
  endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL Clang)
  if(CMAKE_C_COMPILER_VERSION VERSION_LESS ${MINIMUM_CLANG_VERSION})
    message(WARNING
      "Compilation with clang ${CMAKE_C_COMPILER_VERSION} is NOT supported"
    )
  endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL GNU)
  if(CMAKE_C_COMPILER_VERSION VERSION_LESS ${MINIMUM_GNU_VERSION})
    message(WARNING
      "Compilation with gcc ${CMAKE_C_COMPILER_VERSION} is NOT supported"
    )
  endif()
else()
  message(WARNING
    "Compilation with ${CMAKE_C_COMPILER_ID} is NOT supported. Compilation of "
    "project drake_cxx_python may fail."
  )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${MINIMUM_APPLE_CLANG_VERSION})
    message(WARNING
      "Compilation with clang++ ${CMAKE_CXX_COMPILER_VERSION} is NOT supported"
    )
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${MINIMUM_CLANG_VERSION})
    message(WARNING
      "Compilation with clang++ ${CMAKE_CXX_COMPILER_VERSION} is NOT supported"
    )
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${MINIMUM_GNU_VERSION})
    message(WARNING
      "Compilation with g++ ${CMAKE_CXX_COMPILER_VERSION} is NOT supported"
    )
  endif()
else()
  message(WARNING
    "Compilation with ${CMAKE_CXX_COMPILER_ID} is NOT supported. Compilation "
    "of project drake_cxx_python may fail."
  )
endif()

# Determine the CMAKE_BUILD_TYPE. We'll store it as BUILD_TYPE_LOWER so that
# we can treat it as case-insensitive in our string comparisons.
get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(IS_MULTI_CONFIG)
  message(FATAL_ERROR "Drake does not support multi-config generators")
endif()
set(SUPPORTED_BUILD_TYPES Release RelWithDebInfo Debug MinSizeRel)
string(REPLACE ";" " " SUPPORTED_BUILD_TYPES_STRING
  "${SUPPORTED_BUILD_TYPES}"
)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
  STRINGS "${SUPPORTED_BUILD_TYPES}"
)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: ${SUPPORTED_BUILD_TYPES_STRING}"
    FORCE
  )
endif()
string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
string(TOLOWER "${SUPPORTED_BUILD_TYPES}" SUPPORTED_BUILD_TYPES_LOWER)
if(NOT BUILD_TYPE_LOWER IN_LIST SUPPORTED_BUILD_TYPES_LOWER)
  message(WARNING
    "Configuration CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}' is NOT supported. "
    "Defaulting to Release, options are: ${SUPPORTED_BUILD_TYPES_STRING}"
  )
  set(BUILD_TYPE_LOWER release)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: ${SUPPORTED_BUILD_TYPES_STRING}"
    FORCE
  )
endif()

# TODO(jwnimmer-tri) We don't currently pass along the user's selected C++
# standard nor CMAKE_CXX_FLAGS to Bazel, but we should.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

# The supported Python major/minor versions should match those listed in
# doc/_pages/from_source.md and setup/python/pyproject.toml.
if(APPLE)
  # The macOS python version should match what's listed in both the
  # tools/workspace/python/repository.bzl and doc/_pages/installation.md.
  set(SUPPORTED_PYTHON_VERSION 3.13)
else()
  if(UNIX_DISTRIBUTION_CODENAME STREQUAL noble)
    set(SUPPORTED_PYTHON_VERSION 3.12)
  else()  # UNIX_DISTRIBUTION_CODENAME := jammy
    set(SUPPORTED_PYTHON_VERSION 3.10)
  endif()
endif()

# Next we'll very carefully choose which Python interpreter to use.
#
# - If the user provided the legacy spelling -DPYTHON_EXECUTABLE, shift that
#   into -DPython_EXECUTABLE instead and continue (with a warning).
#
# - If the user provided -DPython_EXECUTABLE, take it at face value (and
#   therefore error out if they gave us a broken definition).
#
# - Otherwise, try to find SUPPORTED_PYTHON_VERSION and use it if found.
#
# - Otherwise, try to find any Python 3 interpreter at all.
#
# In all cases, we'll warn in case the found Python is not supported.
if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH
    "Path to the python3 executable" FORCE
  )
  message(WARNING
    "To select a Python interpreter, you should define Python_EXECUTABLE "
    "not PYTHON_EXECUTABLE. The uppercase spelling is used for backwards "
    "compatibility only.")
  unset(PYTHON_EXECUTABLE CACHE)
endif()
if(Python_EXECUTABLE)
  find_package(Python 3 EXACT MODULE REQUIRED
    COMPONENTS Development Interpreter
  )
else()
  find_package(Python ${SUPPORTED_PYTHON_VERSION} EXACT MODULE
    COMPONENTS Development Interpreter
  )
  if(NOT Python_FOUND)
    find_package(Python 3 EXACT MODULE REQUIRED
      COMPONENTS Development Interpreter
    )
  endif()
endif()
if(NOT Python_INTERPRETER_ID STREQUAL Python)
  message(WARNING
    "Python interpreter ${Python_INTERPRETER_ID} is NOT supported. Python "
    "code in project drake_cxx_python may fail at runtime."
  )
endif()
set(PYTHON_VERSION_MAJOR_MINOR
  "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}"
)
if(NOT PYTHON_VERSION_MAJOR_MINOR VERSION_EQUAL SUPPORTED_PYTHON_VERSION)
  message(WARNING
    "The found Python version ${PYTHON_VERSION_MAJOR_MINOR} differs from "
    "Drake's preferred version ${SUPPORTED_PYTHON_VERSION} on this platform. "
    "You may experience compatibility problems that are outside the scope of "
    "Drake's continuous integration test suites."
  )
endif()

set(BAZEL_REPO_ENV)

if(NOT APPLE)
  string(APPEND BAZEL_REPO_ENV
    " --repo_env=CC=${CMAKE_C_COMPILER}"
    " --repo_env=CXX=${CMAKE_CXX_COMPILER}"
  )
  # Pass a Bazel flag which enforces the underlying compiler so that
  # compiler- and version-specific logic (e.g., warnings) is enforced in the
  # build, since rules_cc's support for compiler identification is not as
  # robust as using CMake's CMAKE_<LANG>_COMPILER_ID.
  if(CMAKE_C_COMPILER_ID STREQUAL Clang)
    string(APPEND BAZEL_REPO_ENV
      " --@drake//tools/cc_toolchain:compiler=clang"
    )
  elseif(CMAKE_C_COMPILER_ID STREQUAL GNU)
    string(APPEND BAZEL_REPO_ENV
      " --@drake//tools/cc_toolchain:compiler=gcc"
    )
  endif()
endif()

get_filename_component(PROJECT_BINARY_DIR_REALPATH
  "${PROJECT_BINARY_DIR}" REALPATH
)
get_filename_component(PROJECT_SOURCE_DIR_REALPATH
  "${PROJECT_SOURCE_DIR}" REALPATH
)

# Check whether the PROJECT_BINARY_DIR is a subdirectory of the
# PROJECT_SOURCE_DIR.
string(FIND "${PROJECT_BINARY_DIR_REALPATH}/" "${PROJECT_SOURCE_DIR_REALPATH}/"
  STRING_FIND_RESULT_VARIABLE
)

if(STRING_FIND_RESULT_VARIABLE EQUAL 0)
  # The --output_base cannot be within the WORKSPACE (a subdirectory of
  # PROJECT_SOURCE_DIR), so fallback to the using the same parent directory
  # that Bazel uses by default for its --output_base.
  if(APPLE)
    set(BAZEL_OUTPUT_BASE "/var/tmp")
  else()
    set(BAZEL_OUTPUT_BASE "$ENV{HOME}/.cache/bazel")
  endif()
else()
  set(BAZEL_OUTPUT_BASE "${PROJECT_BINARY_DIR}")
endif()

# Compute the MD5 hash of the PROJECT_BINARY_DIR rather than the WORKSPACE
# (PROJECT_SOURCE_DIR) to avoid colliding with the directory that Bazel uses by
# default.
string(MD5 PROJECT_BINARY_DIR_MD5 "${PROJECT_BINARY_DIR_REALPATH}")
set(BAZEL_OUTPUT_BASE
  "${BAZEL_OUTPUT_BASE}/_bazel_$ENV{USER}/${PROJECT_BINARY_DIR_MD5}"
)

# Symlinks the C++ include path for TARGET as workspace/NAME/include, e.g.
#  workspace/eigen/include -> .../build/install/include/eigen3
function(symlink_external_repository_includes NAME TARGET)
  get_target_property(include_dir ${TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  set(workspace ${CMAKE_CURRENT_BINARY_DIR}/external/workspace)
  file(MAKE_DIRECTORY ${workspace}/${NAME})
  file(CREATE_LINK ${include_dir} ${workspace}/${NAME}/include SYMBOLIC)
endfunction()

# Symlinks the C++ libraries for TARGET as workspace/NAME/lib/*, e.g.
#  workspace/fmt/lib/libfmt.so.6.1.2 -> .../build/install/lib/libfmt.so.6.1.2
#  workspace/fmt/lib/libfmt.so.6 -> .../build/install/lib/libfmt.so.6.1.2
# If USE_SO_MINOR_VERSION is ON, the last line above uses .so.x.y instead, i.e.
#  workspace/fmt/lib/libfmt.so.6.1 -> ../build/install/lib/libfmt.so.6.1.2
function(symlink_external_repository_libs NAME TARGET USE_SO_MINOR_VERSION)
  set(workspace "${CMAKE_CURRENT_BINARY_DIR}/external/workspace")
  file(MAKE_DIRECTORY "${workspace}/${NAME}/lib")
  # Link the full library name (i.e., libfmt.so.6.1.2 in the case of shared).
  get_target_property(location ${TARGET} LOCATION_${CMAKE_BUILD_TYPE})
  if(NOT location)
    message(FATAL_ERROR "Target ${TARGET} has no library in LOCATION_${CMAKE_BUILD_TYPE}")
  endif()
  get_filename_component(basename "${location}" NAME)
  file(CREATE_LINK "${location}" "${workspace}/${NAME}/lib/${basename}" SYMBOLIC)
  # Link the SONAME spelling in case of shared libraries.
  # If the basename does not match this pattern, this part is all a no-op.
  if(USE_SO_MINOR_VERSION)
    string(REGEX REPLACE "(\\.so\\.[0-9]+\\.[0-9]+)\\.[0-9]+$" "\\1"
      other_basename "${basename}")
    string(REGEX REPLACE "(\\.[0-9]+\\.[0-9]+)\\.[0-9]+\\.dylib$" "\\1.dylib"
      other_basename "${other_basename}")
  else()
    string(REGEX REPLACE "(\\.so\\.[0-9]+)\\.[0-9]+\\.[0-9]+$" "\\1"
      other_basename "${basename}")
    string(REGEX REPLACE "(\\.[0-9]+)\\.[0-9]+\\.[0-9]+\\.dylib$" "\\1.dylib"
      other_basename "${other_basename}")
  endif()
  file(CREATE_LINK "${location}" "${workspace}/${NAME}/lib/${other_basename}" SYMBOLIC)
endfunction()

macro(override_module NAME)
  set(local_path "${CMAKE_CURRENT_BINARY_DIR}/external/workspace/${NAME}")
  file(GENERATE OUTPUT "${local_path}/MODULE.bazel"
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/external/workspace/${NAME}/MODULE.bazel.in")
  file(GENERATE OUTPUT "${local_path}/BUILD.bazel"
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/external/workspace/${NAME}/BUILD.bazel.in")
  file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/cmake/external/workspace/conversion.bzl"
    "${local_path}/conversion.bzl"
    SYMBOLIC)
  string(APPEND BAZEL_REPO_ENV
      " --override_module=${NAME}=${local_path}")
  string(APPEND BAZEL_REPO_ENV
      " --@drake//tools/workspace/${NAME}:with_user_${NAME}=True")
endmacro()

macro(override_repository NAME)
  set(local_path "${CMAKE_CURRENT_BINARY_DIR}/external/workspace/${NAME}")
  file(MAKE_DIRECTORY "${local_path}")
  file(GENERATE OUTPUT "${local_path}/MODULE.bazel"
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/external/workspace/${NAME}/MODULE.bazel.in")
  file(GENERATE OUTPUT "${local_path}/BUILD.bazel"
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/external/workspace/${NAME}/BUILD.bazel.in")
  string(APPEND BAZEL_REPO_ENV
      " --override_repository=drake++drake_dep_repositories+${NAME}=${local_path}")
  string(APPEND BAZEL_REPO_ENV
      " --@drake//tools/workspace/${NAME}:with_user_${NAME}=True")
endmacro()

if(APPLE AND DRAKE_CI_ENABLE_PACKAGING)
  # When building a macOS *.tar.gz binary release, never use Homebrew for C++
  # dependencies, because it offers absolutely no version stability.
  set(DEFAULT_WITH_USER_LIBS OFF)
else()
  set(DEFAULT_WITH_USER_LIBS ON)
endif()

option(WITH_USER_EIGEN "Use user-provided Eigen3" ${DEFAULT_WITH_USER_LIBS})

if(WITH_USER_EIGEN)
  find_package(Eigen3 CONFIG REQUIRED)
  symlink_external_repository_includes(eigen Eigen3::Eigen)
  override_module(eigen)
endif()

option(WITH_USER_FMT "Use user-provided fmt" ${DEFAULT_WITH_USER_LIBS})

if(WITH_USER_FMT)
  find_package(fmt CONFIG REQUIRED)
  symlink_external_repository_includes(fmt fmt::fmt)
  symlink_external_repository_libs(fmt fmt::fmt OFF)
  override_module(fmt)
endif()

option(WITH_USER_SPDLOG "Use user-provided spdlog" ${DEFAULT_WITH_USER_LIBS})

if(WITH_USER_SPDLOG)
  if(NOT WITH_USER_FMT)
    message(FATAL_ERROR
      "User-provided spdlog (WITH_USER_SPDLOG) "
      "requires user-provided fmt (WITH_USER_FMT).")
  endif()
  find_package(spdlog CONFIG REQUIRED)
  symlink_external_repository_includes(spdlog spdlog::spdlog)
  # Modern versions of spdlog (i.e. >= 1.11.0) specify a minor version in the
  # SONAME spelling, while older versions only use the major version.
  # We support that logic here to provide compatibility with both.
  # See https://github.com/gabime/spdlog/releases/tag/v1.11.0 for details.
  set(SPDLOG_USE_SO_MINOR_VERSION ON)
  if(${spdlog_VERSION_MAJOR} LESS 1 OR
     (${spdlog_VERSION_MAJOR} EQUAL 1 AND ${spdlog_VERSION_MINOR} LESS 11))
     set(SPDLOG_USE_SO_MINOR_VERSION OFF)
  endif()
  symlink_external_repository_libs(spdlog spdlog::spdlog ${SPDLOG_USE_SO_MINOR_VERSION})
  override_module(spdlog)
endif()

if(NOT APPLE)
  option(WITH_USER_BLAS "Use user-provided BLAS" ${DEFAULT_WITH_USER_LIBS})

  if(WITH_USER_BLAS)
    find_package(BLAS REQUIRED)
    override_repository(blas)
    # The BLAS_LIBRARIES is a CMake list of libraries, where the actual blas
    # is first followed by the transitive closure of libraries it uses. For our
    # purposes, we only want to symlink the primary library. We'll rely on the
    # NEEDED entries on the primary library to link its dependencies.
    list(GET BLAS_LIBRARIES 0 FIRST_BLAS_LIBRARY)
    file(CREATE_LINK
      "${FIRST_BLAS_LIBRARY}"
      "${CMAKE_CURRENT_BINARY_DIR}/external/workspace/blas/libblas.so"
      SYMBOLIC)
  endif()

  option(WITH_USER_LAPACK "Use user-provided LAPACK" ${DEFAULT_WITH_USER_LIBS})

  if(WITH_USER_LAPACK)
    if(NOT WITH_USER_BLAS)
      message(FATAL_ERROR
        "User-provided lapack (WITH_USER_LAPACK) "
        "requires user-provided blas (WITH_USER_BLAS).")
    endif()
    find_package(LAPACK REQUIRED)
    override_repository(lapack)
    # The LAPACK_LIBRARIES is a CMake list of libraries, where the actual lapack
    # is first followed by the transitive closure of libraries it uses. For our
    # purposes, we only want to symlink the primary library. We'll rely on the
    # NEEDED entries on the primary library to link its dependencies.
    list(GET LAPACK_LIBRARIES 0 FIRST_LAPACK_LIBRARY)
    file(CREATE_LINK
      "${FIRST_LAPACK_LIBRARY}"
      "${CMAKE_CURRENT_BINARY_DIR}/external/workspace/lapack/liblapack.so"
      SYMBOLIC)
  endif()
endif()

option(WITH_USER_ZLIB "Use user-provided ZLIB" ${DEFAULT_WITH_USER_LIBS})

if(APPLE)
  # TODO(jwnimmer-tri) Unfortunately we can't use find_package() here because
  # symlink_external_repository_libs() is not compatible with the `zlib.tbd`
  # linker script that CMake finds on macOS. We should work on fixing that
  # routine, and then remove this APPLE-specific logic. When doing so, don't
  # forget to fix our doc/_pages/from_source.md documentation at the same time.
  if(WITH_USER_ZLIB)
    string(APPEND BAZEL_REPO_ENV
        " --@drake//tools/flags:zlib_repo=hardcoded")
  else()
    string(APPEND BAZEL_REPO_ENV
        " --@drake//tools/flags:zlib_repo=source")
  endif()
else()
  if(WITH_USER_ZLIB)
    find_package(ZLIB REQUIRED)
    symlink_external_repository_includes(zlib ZLIB::ZLIB)
    symlink_external_repository_libs(zlib ZLIB::ZLIB OFF)
    override_module(zlib)
  endif()
endif()

set(BAZEL_CONFIG)

# Defines common options for open-source solver dependencies.
# By default, these are all ON.
# Passes the corresponding options to Bazel.
function(open_solver_option SOLVER)
  string(TOUPPER "WITH_${SOLVER}" OPTION_NAME)
  string(TOLOWER "with_${SOLVER}" OPTION_BAZEL_ARG)
  option("${OPTION_NAME}" "Build with support for ${SOLVER}" ON)
  if(${OPTION_NAME})
    string(APPEND BAZEL_CONFIG " --@drake//tools/flags:${OPTION_BAZEL_ARG}=True")
  else()
    string(APPEND BAZEL_CONFIG " --@drake//tools/flags:${OPTION_BAZEL_ARG}=False")
  endif()
  set(BAZEL_CONFIG ${BAZEL_CONFIG} PARENT_SCOPE)
endfunction()

open_solver_option("Clarabel")
open_solver_option("CLP")
open_solver_option("CSDP")
open_solver_option("Ipopt")
open_solver_option("NLopt")
open_solver_option("OSQP")
open_solver_option("SCS")

option(WITH_GUROBI "Build with support for Gurobi" OFF)
if(WITH_GUROBI)
  find_package(Gurobi 10.0 EXACT MODULE REQUIRED)

  string(APPEND BAZEL_CONFIG " --config=gurobi")

  if(NOT APPLE)
    get_filename_component(GUROBI_HOME "${Gurobi_INCLUDE_DIRS}" DIRECTORY)
    string(APPEND BAZEL_REPO_ENV " --repo_env=GUROBI_HOME=${GUROBI_HOME}")
  endif()
endif()

option(WITH_MOSEK "Build with support for MOSEK" OFF)
if(WITH_MOSEK)
  string(APPEND BAZEL_CONFIG " --config=mosek")
endif()

option(WITH_OPENMP "Build with support for OpenMP" OFF)
if(WITH_OPENMP)
  string(APPEND BAZEL_CONFIG " --config=omp")
endif()

set(WITH_ROBOTLOCOMOTION_SNOPT OFF CACHE BOOL
  "Build with support for SNOPT using the RobotLocomotion/snopt private GitHub repository"
)

set(WITH_SNOPT OFF CACHE BOOL
  "Build with support for SNOPT using a SNOPT source archive at SNOPT_PATH"
)

if(WITH_ROBOTLOCOMOTION_SNOPT AND WITH_SNOPT)
  message(FATAL_ERROR
    "WITH_ROBOTLOCOMOTION_SNOPT and WITH_SNOPT options are mutually exclusive"
  )
endif()

if(WITH_ROBOTLOCOMOTION_SNOPT OR WITH_SNOPT)
  enable_language(Fortran)

  if(CMAKE_Fortran_COMPILER_ID STREQUAL GNU)
    if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS ${MINIMUM_GNU_VERSION})
      message(FATAL_ERROR
        "Compilation with gfortran ${CMAKE_Fortran_COMPILER_VERSION} is NOT "
        "supported"
      )
    endif()
  else()
    message(WARNING
      "Compilation with ${CMAKE_Fortran_COMPILER_ID} is NOT supported. "
      "Compilation of project drake_cxx_python may fail."
    )
  endif()

  string(APPEND BAZEL_CONFIG " --config=snopt")

  if(WITH_ROBOTLOCOMOTION_SNOPT)
    string(APPEND BAZEL_REPO_ENV " --repo_env=SNOPT_PATH=git")
  else()
    set(SNOPT_PATH SNOPT_PATH-NOTFOUND CACHE FILEPATH
      "Path to SNOPT source archive"
    )
    if(NOT EXISTS "${SNOPT_PATH}")
      message(FATAL_ERROR
        "SNOPT source archive was NOT found at '${SNOPT_PATH}'"
      )
    endif()
    mark_as_advanced(SNOPT_PATH)
    string(APPEND BAZEL_REPO_ENV " --repo_env=SNOPT_PATH=${SNOPT_PATH}")
  endif()
endif()

if(DRAKE_CI_ENABLE_PACKAGING)
  string(APPEND BAZEL_CONFIG " --config=packaging")
endif()

if(BUILD_TYPE_LOWER STREQUAL debug)
  string(APPEND BAZEL_CONFIG " --config=Debug")
elseif(BUILD_TYPE_LOWER STREQUAL minsizerel)
  string(APPEND BAZEL_CONFIG " --config=MinSizeRel")
elseif(BUILD_TYPE_LOWER STREQUAL release)
  string(APPEND BAZEL_CONFIG " --config=Release")
elseif(BUILD_TYPE_LOWER STREQUAL relwithdebinfo)
  string(APPEND BAZEL_CONFIG " --config=RelWithDebInfo")
endif()

# N.B. If you are testing the CMake API and making changes to `installer.py`,
# you can change this target to something more lightweight, such as
# `//tools/install/dummy:install`.
set(BAZEL_INSTALL_TARGET "@drake//:install")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE STRING
    "Install path prefix, prepended onto install directories" FORCE
  )
endif()

set(BAZEL_INSTALL_ARGS "\${CMAKE_INSTALL_PREFIX}")

if(CMAKE_INSTALL_NAME_TOOL)
  list(INSERT BAZEL_INSTALL_ARGS 0
    "--install_name_tool"
    "${CMAKE_INSTALL_NAME_TOOL}"
  )
endif()

if(CMAKE_STRIP)
  list(INSERT BAZEL_INSTALL_ARGS 0 "--strip_tool" "${CMAKE_STRIP}")
endif()

# If CMAKE_BUILD_TYPE is Debug or RelWithDebInfo, do NOT strip symbols during
# install.
if(BUILD_TYPE_LOWER MATCHES "^(debug|relwithdebinfo)$")
  # SNOPT has restrictions for redistribution given that we are statically
  # linking it in.
  if(WITH_SNOPT OR WITH_ROBOTLOCOMOTION_SNOPT)
    message(WARNING
      "Install configurations Debug and RelWithDebInfo will STILL strip "
      "symbols because support for SNOPT is enabled"
    )
  else()
    list(INSERT BAZEL_INSTALL_ARGS 0 --no_strip)
  endif()
endif()

set(BAZELRC_IMPORT)
foreach(import IN LISTS BAZELRC_IMPORTS)
  string(APPEND BAZELRC_IMPORT "import ${PROJECT_SOURCE_DIR}/${import}\n")
endforeach()

# We need to run Bazel in a dedicated temporary directory. The particular
# name `drake_build_cwd` isn't important, it just needs to be unique. Note,
# however, that the macOS wheel builds also need to know this path, so if it
# ever changes, tools/wheel/macos/build-wheel.sh will also need to be updated.
configure_file(cmake/bazel.rc.in drake_build_cwd/.bazelrc @ONLY)
configure_file(cmake/MODULE.bazel.in drake_build_cwd/MODULE.bazel @ONLY)
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/.bazeliskrc" drake_build_cwd/.bazeliskrc SYMBOLIC)
file(CREATE_LINK "${PROJECT_SOURCE_DIR}/.bazelversion" drake_build_cwd/.bazelversion SYMBOLIC)

find_package(Git)

set(GIT_DIR "${PROJECT_SOURCE_DIR}/.git")

set(GENERATE_DRAKE_VERSION_ARGS)
if(DEFINED DRAKE_VERSION_OVERRIDE)
  list(APPEND GENERATE_DRAKE_VERSION_ARGS
    "-DDRAKE_VERSION_OVERRIDE=${DRAKE_VERSION_OVERRIDE}")
endif()
if(DEFINED DRAKE_GIT_SHA_OVERRIDE)
  list(APPEND GENERATE_DRAKE_VERSION_ARGS
    "-DDRAKE_GIT_SHA_OVERRIDE=${DRAKE_GIT_SHA_OVERRIDE}")
endif()

add_custom_target(drake_version ALL
  COMMAND "${CMAKE_COMMAND}"
    ${GENERATE_DRAKE_VERSION_ARGS}
    "-DGIT_DIR=${GIT_DIR}"
    "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
    "-DINPUT_FILE=${PROJECT_SOURCE_DIR}/tools/install/libdrake/VERSION.TXT.in"
    "-DOUTPUT_FILE=${PROJECT_BINARY_DIR}/VERSION.TXT"
    -P "${PROJECT_SOURCE_DIR}/tools/install/libdrake/generate_version.cmake"
)

add_custom_target(drake_cxx_python ALL
  COMMAND "${Bazel_EXECUTABLE}" build ${BAZEL_INSTALL_TARGET}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/drake_build_cwd"
  USES_TERMINAL
)
add_dependencies(drake_cxx_python drake_version)

install(CODE
  "execute_process(
    COMMAND
      \"${Bazel_EXECUTABLE}\" run ${BAZEL_INSTALL_TARGET}
      -- ${BAZEL_INSTALL_ARGS}
    WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}/drake_build_cwd\"
  )"
  ALL_COMPONENTS
)

install(FILES "${PROJECT_BINARY_DIR}/VERSION.TXT" DESTINATION share/doc/drake)
