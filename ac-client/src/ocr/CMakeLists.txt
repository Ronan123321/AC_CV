add_library(ocr STATIC ocr.h tesseractOcr.h tesseractOcr.cpp)

target_link_libraries(ocr PUBLIC external_deps)

target_include_directories(ocr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
