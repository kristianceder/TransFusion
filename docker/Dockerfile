# ARG PYTORCH="1.10.0"
# ARG CUDA="11.3"
# ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install MMCV, MMDetection and MMSegmentation
# ARG PYTORCH
# ARG CUDA
# ARG MMCV
# ARG MMDET
# ARG MMSEG
# RUN ["/bin/bash", "-c", "pip install --no-cache-dir mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
# RUN pip install --no-cache-dir mmdet==${MMDET} mmsegmentation==${MMSEG}

# # Install MMDetection
# RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d
# WORKDIR /mmdetection3d
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .

# # uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
# RUN pip uninstall pycocotools --no-cache-dir -y
# RUN pip install mmpycocotools --no-cache-dir --force --no-deps

ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"
ENV CUDA_HOME=/usr/local/cuda

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.2.4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
RUN pip install mmdet==2.10.0
RUN pip install mmsegmentation==0.14.1

RUN conda clean --all
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install ipdb
RUN pip install numba==0.53
RUN pip install wandb
#RUN pip install open3d
RUN pip install seaborn
#RUN pip install notebook
RUN pip install ipympl
RUN pip install spconv-cu113

#install mmdetection3d
RUN conda clean --all
COPY . /TransFusion
WORKDIR /TransFusion
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN conda install cython
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps