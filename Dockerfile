FROM continuumio/miniconda3:latest

# Create and activate a new conda environment

ENV PATH /opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA false
ENV DEBIAN_FRONTEND noninteractive

COPY ring_year_env.yaml /tmp/environment.yaml

RUN conda update -n base -c defaults conda && \
    conda init bash && \
    conda config --set auto_activate_base false && \
    conda env create -f /tmp/environment.yaml && \
    conda clean -afy

SHELL ["conda", "run", "-n", "ring_year_env", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

CMD python predict.py

