FROM python:3.6.7-stretch

RUN apt-get update
RUN pip install numpy scipy jupyter wurlitzer
RUN apt-get install -y libhdf5-cpp-100 cifs-utils
RUN mkdir /NAS
RUN mkdir /home/rbry
RUN pip install jupyter_contrib_nbextensions psutil

COPY corrLib.py /usr/local/lib/python3.6/site-packages/
COPY run_command.sh /usr/bin/
COPY dummy_processor.py /
COPY example_nb.ipynb /
RUN chmod +x /usr/bin/run_command.sh
COPY jupyter_config /root/.jupyter
COPY pythonDLLCPU.so /usr/local/lib/

#RUN jupyter nbextensions_configurator enable --user
#RUN jupyter nbextension enable scroll_down/main

ENTRYPOINT ["/usr/bin/run_command.sh"]
CMD [""]
