## Dependencies for CSNN Simulator

current_path=$(pwd)
sudo apt install --yes gcc g++ make cmake libatlas-base-dev libblas-dev libopenblas-dev liblapack-dev liblapacke-dev libopencv-dev

add-apt-repository ppa:rock-core/qt4 && sudo apt install qt4-default

apt install pkg-config libgtk-3-dev libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev \
    libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev \
    python3-numpy libtbb2 libtbb-dev libdc1394-22-dev


mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd ~/opencv_build/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF ..

nproc

make -j40

sudo make install

## Return to the original path before continuing
cd "$current_path"

## Get a fresh copy of the simulator and build it
git clone https://github.com/cosminneamtiu02/csnn_simulator

mkdir -p build
cd build

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:~/opencv_build/opencv/build/

# Fix the path to use csnn_simulator (with underscore) instead of csnn-simulator (with dash)
cmake ../csnn_simulator -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO

make