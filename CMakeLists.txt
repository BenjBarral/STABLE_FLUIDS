CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(OpenCV_DIR /usr/local/share/OpenCV)
FIND_PACKAGE(OpenCV REQUIRED)


SET(Eigen3_DIR /usr/local/share/eigen3/cmake)
find_package (Eigen3 3.3 REQUIRED)

PROJECT(stable_fluids)

# Inform the paths of the libraries depending on your installation
INCLUDE_DIRECTORIES(/usr/local/include)
LINK_DIRECTORIES(/usr/local/lib)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

ADD_EXECUTABLE(fluid-simulator main.cpp)
TARGET_LINK_LIBRARIES(fluid-simulator ${OpenCV_LIBS} Eigen3::Eigen)
