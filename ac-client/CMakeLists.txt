cmake_minimum_required(VERSION 3.11)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(shop_capture VERSION 1)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libs/opencv-local/")

find_package(OpenCV CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(Leptonica REQUIRED IMPORTED_TARGET lept)
pkg_check_modules(Tesseract REQUIRED IMPORTED_TARGET tesseract)
pkg_check_modules(CURL REQUIRED IMPORTED_TARGET libcurl)
pkg_check_modules(ARCHIVE REQUIRED IMPORTED_TARGET libarchive)

add_library(external_deps INTERFACE)

target_link_libraries(external_deps INTERFACE ${OpenCV_LIBS} PkgConfig::Tesseract PkgConfig::Leptonica PkgConfig::CURL PkgConfig::ARCHIVE)

add_subdirectory(src/)
add_subdirectory(src/ocr/)
add_subdirectory(src/elements/)
