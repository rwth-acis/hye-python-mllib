FROM openjdk:14-jdk-slim-buster

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget python3 python3-pip gcc libc-dev python3-dev g++ cmake make
RUN python3 -m pip install --upgrade pip
RUN useradd -ms /bin/bash newuser

COPY --chown=newuser:newuser . /app
WORKDIR /app

# run the rest as unprivileged user
USER newuser
RUN cc -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -fPIC -shared -o libwordcenter.so word_center.c
RUN pip3 install -r required.txt
RUN mkdir models

ENTRYPOINT ["./docker-entrypoint.sh"]
