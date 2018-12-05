FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN mkdir /root/_INSTALL

WORKDIR /root/_INSTALL

RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list
 
ENV TZ Asia/Tokyo
RUN apt-get update \
  && apt-get install -y tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && echo "${TZ}" > /etc/timezone \
  && rm /etc/localtime \
  && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
  && dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        wget \
        vim \
        git \
        net-tools \
        iputils-ping \
        netcat \
        tcpdump \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libboost-dev \
        libprotobuf9v5 \
        libtool \
        protobuf-compiler \
        libprotobuf-dev \
        pkg-config \
        python \
        python-dev \
        python-wheel \
        python-mock \
        rsync \
        zip \
        unzip \
        g++ \
        gfortran \
        mlocate \
        libcupti-dev \
        language-pack-ja-base \
        language-pack-ja \
        tree \
        apt-transport-https \
        xz-utils \
        file \
        && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* && \
        updatedb

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        numpy \
        matplotlib \
        pandas \
        futures \
        flask \
        nose \
        tqdm \
        jedi \
        flake8 \
        autopep8 \
        && \
    python -m ipykernel.kernelspec

# MeCab
RUN curl --insecure -L "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE" -o mecab-0.996.tar.gz \
 && tar xzf mecab-0.996.tar.gz \
 && cd ./mecab-0.996 \
 && ./configure \
 && make \
 && make check \
 && make install \
 && ldconfig
 
# MeCab-python
RUN curl --insecure -L "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7UlJpaWJKM01KRVE" -o mecab-python-0.996.tar.gz \
 && tar xzf mecab-python-0.996.tar.gz \
 && cd ./mecab-python-0.996 \
 && python setup.py build \
 && python setup.py install
 
# IPADic
RUN curl --insecure -L "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7MWVlSDBCSXZMTXM" -o mecab-ipadic-2.7.0-20070801.tar.gz \
 && tar xzf mecab-ipadic-2.7.0-20070801.tar.gz \
 && cd ./mecab-ipadic-2.7.0-20070801 \
 && ./configure --with-charset=utf8 \
 && make \
 && make install
 
# JumanDic
RUN curl --insecure -L "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7X2pESGlLREpxdXM" -o mecab-jumandic-7.0-20130310.tar.gz \
 && tar xzf mecab-jumandic-7.0-20130310.tar.gz \
 && cd mecab-jumandic-7.0-20130310 \
 && ./configure \
 && make \
 && make install
 
# UniDic
RUN curl --insecure -L "https://ja.osdn.net/frs/redir.php?m=ymu&f=%2Funidic%2F58338%2Funidic-mecab-2.1.2_src.zip" -o unidic-mecab-2.1.2_src.zip \
 && unzip unidic-mecab-2.1.2_src.zip \
 && cd unidic-mecab-2.1.2_src \
 && ./configure \
 && make \
 && make install
 
# IPADic-NEologd
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
 && cd mecab-ipadic-neologd \
 && ./bin/install-mecab-ipadic-neologd -n -a -y
 
# UniDic-NEologd
RUN git clone --depth 1 https://github.com/neologd/mecab-unidic-neologd.git \
 && cd mecab-unidic-neologd \
 && sed -ri 's/sudo //g' ./libexec/install-mecab-unidic-neologd.sh \
 && ./bin/install-mecab-unidic-neologd -n -y
 
# Juman
RUN curl --insecure -L "http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/juman/juman-7.01.tar.bz2&name=juman-7.01.tar.bz2" -o juman-7.01.tar.bz2 \
 && tar xjf juman-7.01.tar.bz2 \
 && cd juman-7.01 \
 && ./configure \
 && make \
 && make install \
 && ldconfig
 
# Juman++
RUN curl --insecure -L "http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://lotus.kuee.kyoto-u.ac.jp/nl-resource/jumanpp/jumanpp-1.02.tar.xz&name=jumanpp-1.02.tar.xz" -o jumanpp-1.02.tar.xz \
 && tar xJf jumanpp-1.02.tar.xz \
 && cd jumanpp-1.02 \
 && ./configure \
 && make \
 && make install
 
# Kytea
RUN curl -O http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz \
 && tar xzf kytea-0.4.7.tar.gz \
 && cd kytea-0.4.7 \
 && ./configure \
 && make \
 && make install
 
# Sentencepieces
RUN apt update && apt install -y cmake
RUN git clone https://github.com/google/sentencepiece.git \
 && cd sentencepiece \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make -j $(nproc) \
 && make install \
 && ldconfig -v

# Install Java8
#RUN add-apt-repository ppa:webupd8team/java && \
#    echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
#    apt-get update && apt-get install -y oracle-java8-installer

# Install OpenJDK 8
RUN apt-get update \
 && apt-get install -y openjdk-8-jdk

# Install Bazel for building tensorflow
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-installer-linux-x86_64.sh && \
    chmod +x bazel-0.4.5-installer-linux-x86_64.sh && \
    ./bazel-0.4.5-installer-linux-x86_64.sh

# Copy cuDNN
COPY cudnn-8.0-linux-x64-v6.0.solitairetheme8 /tmp/

RUN tar xzf /tmp/cudnn-8.0-linux-x64-v6.0.solitairetheme8 && \
    cp -a cuda/include/* /usr/local/cuda/include/ && \
    cp -a cuda/lib64/* /usr/local/cuda/lib64/

# Install Tensorflow from source
RUN git clone https://github.com/tensorflow/tensorflow && \
    cd tensorflow && \
    git checkout v1.3.0

COPY workspace.bzl tensorflow/tensorflow/

RUN cd tensorflow && \
    export PYTHON_BIN_PATH=$(which python) && \
    export PYTHON_LIB_PATH="/usr/lib/python2.7/dist-packages" && \
    export TF_NEED_MKL="1" && \
    export TF_DOWNLOAD_MKL="1" && \
    export CC_OPT_FLAGS="-march=native" && \
    export TF_NEED_JEMALLOC="1" && \
    export TF_NEED_GCP="0" && \
    export TF_NEED_HDFS="0" && \
    export TF_NEED_OPENCL="0" && \
    export TF_NEED_S3="0" && \
    export TF_NEED_CUDA="1" && \
    export TF_ENABLE_XLA="1" && \
    export TF_NEED_GDR="1" && \
    export TF_NEED_VERBS="0" && \
    export TF_CUDA_CLANG="0" && \
    export TF_CUDA_VERSION="8.0" && \
    export CUDA_TOOLKIT_PATH="/usr/local/cuda" && \
    export TF_CUDNN_VERSION="6" && \
    export CUDNN_INSTALL_PATH="/usr/local/cuda" && \
    export TF_CUDA_COMPUTE_CAPABILITIES="6.1" && \
    export GCC_HOST_COMPILER_PATH=$(which gcc) && \
    export TF_NEED_MPI="0" && \
    ./configure && \
    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-msse4.1 --copt=-msse4.2 --copt=-mfma --config=cuda //tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

RUN pip install /tmp/tensorflow_pkg/tensorflow-1.3.0-cp27-cp27mu-linux_x86_64.whl

RUN echo '# LANG' >> ~/.bashrc \
 && echo 'export LANG=ja_JP.UTF-8' >> ~/.bashrc

WORKDIR /root
