cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON -D WITH_TBB=ON \
      -D WITH_V4L=ON -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_TESTS=OFF \
      -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
      -D MKL_ROOT_DIR=/opt/intel/mkl \
      -D PYTHON_EXECUTABLE=/usr/bin/python3.6 \
      -D PYTHON_INCLUDE_DIR=/usr/include/python3.6m \
      -D PYTHON_PACKAGES_PATH=/usr/local/lib/python3.6/dist-packages/ \
      ..
