# Add each source and header individually
add_library(elements STATIC
            ImageRegion.h ImageRegion.cpp
            shopsplit.h shopsplit.cpp
            buttons.h buttons.cpp
            champnames.h champnames.cpp
            champslots.h champslots.cpp
            gold.h gold.cpp
            levelinfo.h levelinfo.cpp)

target_link_libraries(elements PUBLIC external_deps)
target_link_libraries(elements PRIVATE ocr)

target_include_directories(elements PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(element_testing testing.cpp)
target_link_libraries(element_testing elements ocr)
