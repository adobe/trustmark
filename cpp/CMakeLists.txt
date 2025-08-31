cmake_minimum_required(VERSION 3.16)
project(TrustMarkCpp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime (avoid schema conflicts by not using find_package)
# Instead, manually specify the paths to avoid ONNX library conflicts

# Find OpenCV for image processing
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/trustmark)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/ort/include)

# Create the library
add_library(trustmark_cpp
    trustmark/trustmark.cpp
    trustmark/onnx_session.cpp
    trustmark/image_processor.cpp
    trustmark/bch_ecc.cpp
)

# Link libraries
target_link_libraries(trustmark_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ort/lib/libonnxruntime.dylib
    ${OpenCV_LIBS}
)

# Set properties
set_target_properties(trustmark_cpp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "trustmark/trustmark.h"
)

# Create example executable
add_executable(trustmark_example
    examples/example.cpp
)

target_include_directories(trustmark_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trustmark_example trustmark_cpp)

# Installation
install(TARGETS trustmark_cpp
    EXPORT TrustMarkCppTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

install(DIRECTORY trustmark/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Export targets
install(EXPORT TrustMarkCppTargets
    FILE TrustMarkCppTargets.cmake
    NAMESPACE TrustMarkCpp::
    DESTINATION lib/cmake/TrustMarkCpp
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TrustMarkCppConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TrustMarkCppConfig.cmake"
    INSTALL_DESTINATION lib/cmake/TrustMarkCpp
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TrustMarkCppConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TrustMarkCppConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TrustMarkCppConfigVersion.cmake"
    DESTINATION lib/cmake/TrustMarkCpp
)
# Link against local libonnxruntime
set(ORT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ort/lib/libonnxruntime.dylib)
if(NOT EXISTS ${ORT_LIB})
  message(FATAL_ERROR "libonnxruntime not found at ${ORT_LIB}")
endif()

# Ensure rpath includes our third_party lib
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "@loader_path/../third_party/ort/lib;@loader_path/../lib")
set(CMAKE_BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ort/lib")

# Link
set(ADDITIONAL_LIBS ${ORT_LIB})

# Apply to our target
# Replace previous target_link_libraries call

# Link libraries
 target_link_libraries(trustmark_cpp
     ${ADDITIONAL_LIBS}
     ${OpenCV_LIBS}
 )