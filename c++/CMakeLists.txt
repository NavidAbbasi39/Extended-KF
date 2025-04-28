cmake_minimum_required(VERSION 3.10)
project(EKFRobotArm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 for linear algebra
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find matplotlib-cpp for plotting (optional)
# You can comment this out if you don't want plotting
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

add_subdirectory(external/matplotlib-cpp) # Optional, see note below

include_directories(include)

add_executable(EKFRobotArm
    src/main.cpp
    src/ExtendedKalmanFilter.cpp
    src/PDController.cpp
    src/RobotArm.cpp
    src/Simulation.cpp
)

target_link_libraries(EKFRobotArm
    Eigen3::Eigen
    # matplotlib-cpp # Uncomment if using matplotlib-cpp
    Python3::Python
    Python3::NumPy
)
