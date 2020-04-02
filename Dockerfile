FROM python:3.7.3-stretch

LABEL maintainer="superhzf@hotmail.com"
LABEL service="homecreditdefault"

# https://www.kaggle.com/getting-started/45288
ENV JOBLIB_TEMP_FOLDER=/tmp

#COPY poetry.lock pyproject.toml ${APP_DIR}/
COPY pyproject.toml ${APP_DIR}/

#RUN apk add --update alpine-sdk wget postgresql-client lz4-dev git libzmq zeromq-dev zeromq
#RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
#RUN source $HOME/.poetry/env && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

RUN apt-get update \
 && for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done \
 && apt-get install -y build-essential wget postgresql-client liblz4-tool git nano \
 && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
RUN /bin/bash -c "source $HOME/.poetry/env; poetry config virtualenvs.create false; poetry install --no-interaction --no-ansi"


ARG APP_NAME
ARG APP_VERSION

ENV APP_NAME ${APP_NAME}
ENV APP_VERSION ${APP_VERSION}

# jupyterlab set kernel
RUN python -m ipykernel.kernelspec

# set up our notebook config.
COPY jupyter_notebook_config_py3.py /root/.jupyter/
RUN mv /root/.jupyter/jupyter_notebook_config_py3.py /root/.jupyter/jupyter_notebook_config.py

COPY run_jupyter.sh /
RUN chmod +x /run_jupyter.sh

# set up pudb
ENV PYTHONBREAKPOINT pudb.set_trace

# dockerize:https://github.com/jwilder/dockerize
ENV DOCKERIZE_VERSION v0.6.1
RUN wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz

WORKDIR /usr/src/app

# setup pythonpath
ENV PYTHONPATH="/usr/src/app"