cmake_minimum_required(VERSION 3.10.2)
project(calibrator)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
link_libraries(
    ${OpenCV_LIBS}
)

add_executable(app src/app.cpp)
add_executable(convert src/convertor.cpp)