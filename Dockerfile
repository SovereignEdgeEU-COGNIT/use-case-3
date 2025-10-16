FROM ubuntu:22.04

RUN apt update && apt install -y python3.10 build-essential gcc cmake git python3.10-venv

RUN git clone https://github.com/SovereignEdgeEU-COGNIT/use-case-3

WORKDIR /use-case-3
RUN git status && git pull
RUN python3.10 -m venv _venv && _venv/bin/pip3.10 install -r requirements.txt

COPY cognit.yml cognit.yml

CMD ["bash"]
