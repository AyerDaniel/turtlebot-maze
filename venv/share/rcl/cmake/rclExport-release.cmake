#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "rcl::rcl" for configuration "Release"
set_property(TARGET rcl::rcl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(rcl::rcl PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "rcl_logging_spdlog::rcl_logging_spdlog;service_msgs::service_msgs__rosidl_generator_c;service_msgs::service_msgs__rosidl_typesupport_fastrtps_c;service_msgs::service_msgs__rosidl_typesupport_fastrtps_cpp;service_msgs::service_msgs__rosidl_typesupport_introspection_c;service_msgs::service_msgs__rosidl_typesupport_c;service_msgs::service_msgs__rosidl_typesupport_introspection_cpp;service_msgs::service_msgs__rosidl_typesupport_cpp;service_msgs::service_msgs__rosidl_generator_py;tracetools::tracetools"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librcl.so"
  IMPORTED_SONAME_RELEASE "librcl.so"
  )

list(APPEND _cmake_import_check_targets rcl::rcl )
list(APPEND _cmake_import_check_files_for_rcl::rcl "${_IMPORT_PREFIX}/lib/librcl.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
