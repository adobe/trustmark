cmake_minimum_required(VERSION 3.16)
project(trustmark_cpp LANGUAGES CXX)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation directory")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime headers and library are required. Set ONNXRUNTIME_ROOT to its installation directory.")
endif()

add_executable(trustmark-runner src/main.cpp src/bch.cpp)
target_compile_features(trustmark-runner PRIVATE cxx_std_17)
target_include_directories(trustmark-runner PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustmark-runner PRIVATE "${ONNXRUNTIME_LIBRARY}" ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
    add_executable(trustmark-bch-test tests/bch.cpp src/bch.cpp)
    target_compile_features(trustmark-bch-test PRIVATE cxx_std_17)
    target_include_directories(trustmark-bch-test PRIVATE src)
    if(MSVC)
        target_compile_options(trustmark-bch-test PRIVATE /UNDEBUG)
    else()
        target_compile_options(trustmark-bch-test PRIVATE -UNDEBUG)
    endif()
    add_test(NAME trustmark-bch COMMAND trustmark-bch-test)
endif()

if(UNIX)
    get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    set_target_properties(trustmark-runner PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
endif()
