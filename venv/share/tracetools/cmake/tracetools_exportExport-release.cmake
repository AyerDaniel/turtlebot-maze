#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tracetools::tracetools" for configuration "Release"
set_property(TARGET tracetools::tracetools APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tracetools::tracetools PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtracetools.so"
  IMPORTED_SONAME_RELEASE "libtracetools.so"
  )

list(APPEND _cmake_import_check_targets tracetools::tracetools )
list(APPEND _cmake_import_check_files_for_tracetools::tracetools "${_IMPORT_PREFIX}/lib/libtracetools.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
