FROM ubuntu:14.04

ENV TERM=xterm
ENV LANG en_US.UTF-8

# install scikit-learn prereqs
RUN apt-get update && \
    apt-get -y install build-essential python python-pip python-dev python-setuptools python-numpy python-scipy libopenblas-dev python-matplotlib python-pandas python-nose
    
#RUN python -c 'import numpy; numpy.test("full");' && \
#    python -c 'import scipy; scipy.test("full");' && \
#    nosetests pandas

# use blas for speedups in some scikit-learn modules
RUN update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3 && \
    update-alternatives --set liblapack.so.3 /usr/lib/lapack/liblapack.so.3

# install latest scikit-learn
RUN pip install -U scikit-learn && nosetests -v sklearn

# install and verify other project dependencies
RUN pip install prettytable pytest PyYAML yamlordereddictloader

# madness project can be shared here via a volume
WORKDIR /madness

CMD py.test
