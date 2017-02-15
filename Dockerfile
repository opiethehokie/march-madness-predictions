FROM ubuntu:16.04

ENV TERM xterm
ENV LANG en_US.UTF-8  
ENV LC_ALL C

ENV MPLBACKEND agg

RUN apt-get update && \
    apt-get -y install build-essential \
                       libopenblas-dev \
                       llvm-3.8-dev \
                       pandoc \
                       python3 \
                       python3-dev \
                       python3-nose \
                       python3-numexpr \
                       python3-numpy \
                       python3-pip \
                       python3-setuptools \
                       python3-scipy \
                       python3-tk \
                       zlib1g-dev && \
    update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3 && \
    update-alternatives --set liblapack.so.3 /usr/lib/lapack/liblapack.so.3 && \
    rm -rf /var/lib/apt/lists/*

RUN LLVM_CONFIG=/usr/bin/llvm-config-3.8 python3 -m pip --no-cache-dir install llvmlite numba

RUN python3 -m pip --no-cache-dir install bottleneck \
                                          matplotlib \
                                          pandas \
                                          pylint \
                                          pypandoc \
                                          pytest \
                                          scikit-learn \
                                          mlxtend

#RUN python3 -c 'import bottleneck; bottleneck.test();' && \
#    python3 -c 'import numexpr; numexpr.test();' && \
#    python3 -c 'import numpy; numpy.test();' && \
#    python3 -c 'import pandas; pandas.test();'
#    python3 -c 'import scipy; scipy.test();' && \
#    nosetests3 sklearn

RUN python3 -m pip --no-cache-dir install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl && \
    python3 -c 'import tensorflow;'

RUN python3 -m pip --no-cache-dir install pyyaml \
                                          yamlordereddictloader

WORKDIR /workdir

CMD echo "ML stack installation complete"
