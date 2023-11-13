FROM python:3.8

RUN python3.8 -m pip install --upgrade pip

COPY requirements.txt /supreme/requirements.txt
COPY requirements-dev.txt /supreme/requirements-dev.txt
COPY setup.py /supreme/setup.py

WORKDIR /supreme

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
RUN pip install -r /supreme/requirements.txt
RUN pip install -r /supreme/requirements-dev.txt

COPY . /supreme

ENV SUPREME=/supreme/lib/supreme/src
ENV PYTHONPATH=${SUPREME}:${PYTHONPATH}

CMD [ "python", "project/supreme/src/main.py"]