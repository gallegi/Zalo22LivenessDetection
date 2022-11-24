FROM t4e_liveness_base:latest

USER root

RUN mkdir /code

COPY start_jupyter.sh /code

CMD ["/bin/bash /code/start_jupyter.sh"]