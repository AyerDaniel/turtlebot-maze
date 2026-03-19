#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "rcl_logging_spdlog::rcl_logging_spdlog" for configuration "Release"
set_property(TARGET rcl_logging_spdlog::rcl_logging_spdlog APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(rcl_logging_spdlog::rcl_logging_spdlog PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "rcpputils::rcpputils;rcutils::rcutils;spdlog::spdlog"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librcl_logging_spdlog.so"
  IMPORTED_SONAME_RELEASE "librcl_logging_spdlog.so"
  )

list(APPEND _cmake_import_check_targets rcl_logging_spdlog::rcl_logging_spdlog )
list(APPEND _cmake_import_check_files_for_rcl_logging_spdlog::rcl_logging_spdlog "${_IMPORT_PREFIX}/lib/librcl_logging_spdlog.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
