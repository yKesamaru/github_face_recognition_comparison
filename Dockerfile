# base image
FROM python:3.8.12
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# -----------------------------------
# create required folder
RUN mkdir -p /app && chown -R 1001:0 /app
RUN mkdir -p /app/deepface
RUN mkdir -p /root/.deepface/weights  # weightsフォルダのコピー先を作成

# -----------------------------------
# switch to application directory
WORKDIR /app

# -----------------------------------
# update image OS and install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------
# Install CUDA and cuDNN
RUN distribution=$( . /etc/os-release; echo $ID$VERSION_ID) && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-5 libcudnn8 libcudnn8-dev libcublas-12-0 && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH=/usr/local/cuda-11.5/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# -----------------------------------
# Copy required files from repo into image
COPY ./deepface /app/deepface
COPY ./requirements.txt /app/requirements.txt
COPY ./requirements_local /app/requirements_local.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/
COPY ./entrypoint.sh /app/deepface/api/src/entrypoint.sh

# Copy weights folder into container
COPY ./weights /root/.deepface/weights

# -----------------------------------
# install dependencies - deepface with these dependency versions is working
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local.txt
# install deepface from source code (always up-to-date)
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .

# Install TensorFlow compatible with CUDA setup
RUN pip uninstall -y tensorflow keras && \
    pip install --no-cache-dir tensorflow==2.12 && \
    pip install matplotlib && \
    pip install scikit-learn

# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1

# -----------------------------------
# set working directory for runtime
WORKDIR /app/deepface
