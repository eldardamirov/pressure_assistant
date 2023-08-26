FROM ubuntu:22.10

ENV DEBIAN_FRONTEND=noninteractive LANG=C TZ=UTC
ENV TERM linux

# install some basic utilities
RUN set -xue ;\
    apt-get update ;\
    apt-get install -y --no-install-recommends \
        build-essential \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0 \
        wget \
        python3-dev \
        python3-pip \
    ;\
    rm -rf /var/lib/apt/lists/*

# install libs and frameworks
RUN pip3 install setuptools ;\
    pip3 install numpy ;\
    pip3 install pandas ;\
    pip3 install matplotlib ;\
    pip3 install torch torchvision; \
    # pip3 install tensorflow; \
    pip3 install jupyterlab; \
    pip3 install llama-cpp-python==0.1.78; \
    pip3 install langchain==0.0.174; \
    pip3 install huggingface-hub==0.14.1; \
    pip3 install chromadb==0.3.23; \
    pip3 install pdfminer.six==20221105; \
    pip3 install unstructured==0.6.10; \
    pip3 install gradio==3.32.0; \
    pip3 install tabulate
    
    
COPY . .
    
    
CMD ["python3", "./app.py"]





# WORKDIR /home/edamirov/Development/SPB_hack/make_docker/saiga_13b_llamacpp_retrieval_qa
# WORKDIR /playground

# run the command
# CMD ["/bin/bash"]
# CMD ["jupyter", "notebook", "--port=8789", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
# CMD ["python3", "app.py"]

# ENTRYPOINT [ "python3" ]
# CMD [ "export","FLASK_APP=run.py" ]
# CMD [ "set", "FLASK_APP=run.py" ]
# CMD [ "flask", "run", "--host=0.0.0.0" ]
