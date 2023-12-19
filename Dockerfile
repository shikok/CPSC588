FROM python:3.10 
ARG USER_ID
ARG GROUP_ID

RUN apt-get update
RUN apt-get install -y --no-install-recommends
# Need to agree to install options, hence printf piped to apt-get
RUN printf "Y" | apt-get install texlive texstudio texlive-latex-extra texlive-fonts-recommended dvipng cm-super

SHELL ["/bin/bash", "-c"]
WORKDIR /home/joe/spatial_omics_reimp

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-index pyg_lib -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
&& pip install --no-index torch_scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
&& pip install --no-index torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
&& pip install --no-index torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
&& pip install --no-index torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
&& pip install torch-geometric 

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
