add_library(shop STATIC shop.h shop.cpp getwindow.cpp overlayrect.cpp "overlayrect.h" "WINDOW.h")

target_include_directories(shop PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shop PUBLIC elements ocr external_deps)

add_executable(run_main WIN32 main.cpp "WINDOW.h")
add_executable(test_main test_main.cpp "WINDOW.h")

target_link_libraries(run_main PRIVATE shop)
target_link_libraries(test_main PRIVATE shop)

#target_include_directories(run_main PUBLIC ${CMAKE_SOURCE_DIR}/ocr)

#list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libs/opencv-local/")
#message("cmake prefix path: ${CMAKE_PREFIX_PATH}")

#message(STATUS "OpenCV include dir: " ${OpenCV_INCLUDE_DIRS})
#message(STATUS "Leptonica include dir: " ${Leptonica_INCLUDEDIR}/leptonica/)
#message(STATUS "Tesseract include dir: " ${Tesseract_INCLUDEDIR}/tesseract/)
#target_include_directories(run_main PRIVATE ${OpenCV_INCLUDE_DIRS} ${Leptonica_INCLUDEDIR}/leptonica/ ${Tesseract_INCLUDEDIR}/tesseract/)

#message(STATUS "OpenCV lib dir: $ENV{HOME}/libs/opencv-local/lib/")
#message(STATUS "Leptonica lib dir: " ${Leptonica_LIBRARY_DIRS})
#message(STATUS "Tesseract lib dir: " ${Tesseract_LIBRARY_DIRS})
#target_link_directories(run_main PRIVATE "$ENV{HOME}/libs/opencv-local/lib/" ${Leptonica_LIBRARY_DIRS} ${Tesseract_LIBRARY_DIRS})

#target_link_libraries(run_main PRIVATE ${OpenCV_LIBS} PkgConfig::Tesseract PkgConfig::Leptonica PkgConfig::CURL)
