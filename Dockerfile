FROM namnt42/t4e_liveness_base:latest

USER root

RUN mkdir /code
RUN mkdir /result
RUN mkdir /data

COPY start_jupyter.sh /code

COPY configs/* /code/configs/
COPY src/* /code/src/

COPY train.py /code/
COPY predict* /code/
COPY evaluate* /code/
COPY predict_notebook.ipynb /code/

COPY models/v4_metric_learning_regnet_y_16gf/fold0/* /code/models/v4_metric_learning_regnet_y_16gf/fold0/
COPY models/cspdarknet_lstm/* /code/models/cspdarknet_lstm/

