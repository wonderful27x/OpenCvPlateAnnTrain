
# CMakeList.txt: AnnPlateTrain 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

include_directories("E:/OPENCV/opencv4.1.2/INSTALL/opencv/build/include")

link_directories("E:/OPENCV/opencv4.1.2/INSTALL/opencv/build/x64/vc15/lib")

file(GLOB SRC *.cpp)

# 将源代码添加到此项目的可执行文件。
add_executable (AnnPlateTrainProject ${SRC})

target_link_libraries(AnnPlateTrainProject opencv_world412d)

# TODO: 如有需要，请添加测试并安装目标。
