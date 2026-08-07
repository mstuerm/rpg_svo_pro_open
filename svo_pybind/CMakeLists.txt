project(svo_pybind)
cmake_minimum_required(VERSION 3.5)

find_package(catkin_simple REQUIRED)
catkin_simple()
find_package(OpenCV REQUIRED)
find_package(PythonInterp REQUIRED)

# FindPythonInterp and FindPythonLibs can independently select different
# Python installations (e.g. a 3.10 venv plus Ubuntu's 3.8 development
# files). Ask the selected interpreter for its own development paths before
# invoking FindPythonLibs.
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE SVO_PYTHON_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import os, sysconfig; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))"
  OUTPUT_VARIABLE SVO_PYTHON_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
set(PYTHON_INCLUDE_DIR "${SVO_PYTHON_INCLUDE_DIR}" CACHE PATH
    "Python include directory" FORCE)
set(PYTHON_LIBRARY "${SVO_PYTHON_LIBRARY}" CACHE FILEPATH
    "Python library" FORCE)
find_package(PythonLibs REQUIRED)

# catkin/FindPythonInterp uses the legacy all-uppercase variables, while
# recent pybind11 uses CMake's FindPython module and its mixed-case variables.
# Pin both to the same interpreter to avoid silently mixing Ubuntu's Python
# 3.8 with the project's Python 3.10 virtual environment.
set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH
    "Python interpreter used by pybind11" FORCE)
set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIRS}" CACHE PATH
    "Python include directory used by pybind11" FORCE)
set(Python_LIBRARY "${PYTHON_LIBRARIES}" CACHE FILEPATH
    "Python library used by pybind11" FORCE)
set(PYBIND11_FINDPYTHON ON)

# pybind11 is installed in the project's Python virtual environment. Its
# CMake package is not necessarily on catkin's CMAKE_PREFIX_PATH, so ask the
# same interpreter selected by catkin where the package configuration lives.
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir
  RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
  OUTPUT_VARIABLE PYBIND11_CMAKEDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(PYBIND11_CMAKEDIR_RESULT EQUAL 0 AND EXISTS "${PYBIND11_CMAKEDIR}")
  set(pybind11_DIR "${PYBIND11_CMAKEDIR}" CACHE PATH
      "pybind11 CMake package directory" FORCE)
  find_package(pybind11 CONFIG REQUIRED)
else()
  set(PYBIND11_SOURCE_DIR
      "$ENV{FLIGHTMARE_PATH}/flightlib/externals/pybind11-src")
  if(EXISTS "${PYBIND11_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "Using vendored pybind11: ${PYBIND11_SOURCE_DIR}")
    add_subdirectory(
      "${PYBIND11_SOURCE_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
      EXCLUDE_FROM_ALL
    )
  else()
    message(FATAL_ERROR
      "pybind11 was not found in Python and vendored source is unavailable")
  endif()
endif()
find_package(yaml-cpp REQUIRED)

include(SvoSetup)
include_directories(${PYTHON_INCLUDE_DIRS})

pybind11_add_module(svo_pybind MODULE src/svo_pybind.cpp)

# The exported SVO catkin libraries are absolute paths and already contain
# the complete transitive SVO/vikit dependency set. Avoid adding the same
# libraries as bare -l names (catkin does not export a matching -L flag).
# Also discard ROS Noetic's Python 3.8 library: extension modules resolve the
# Python API from the Python 3.10 interpreter loading them.
set(SVO_PYBIND_CATKIN_LIBRARIES)
foreach(LIBRARY ${catkin_LIBRARIES})
  if(NOT "${LIBRARY}" MATCHES "libpython3\\.8")
    list(APPEND SVO_PYBIND_CATKIN_LIBRARIES "${LIBRARY}")
  endif()
endforeach()
target_link_libraries(svo_pybind PRIVATE
  ${OpenCV_LIBRARIES}
  yaml-cpp
  ${SVO_PYBIND_CATKIN_LIBRARIES}
)

set_target_properties(svo_pybind PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  LIBRARY_OUTPUT_DIRECTORY "${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}"
)

install(TARGETS svo_pybind
  LIBRARY DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}
)
