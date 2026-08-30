FROM nvcr.io/nvidia/pytorch:23.01-py3

USER root
ARG UID=1000
ARG GID=1000
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install the remaining dependencies
# RUN pip install pyeda seaborn imageio scikit-image
# RUN pip uninstall pyparsing -y
# RUN pip install pyparsing==2.4.2


RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN pip install cython ftfy timm
RUN pip install -U openmim
RUN mim install mmcv-full==1.6.2
RUN pip install mmsegmentation==0.27.0

RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install open_clip_torch
RUN pip install einops
RUN pip install transformers==4.19.2
RUN pip install -U transformers
RUN pip install compositional_explanations
RUN pip install wn

