#!/bin/bash
if [ -n "$SERVICE_PORT" ]; then
    sed -i "s/8000/${SERVICE_PORT}/" Config.py
fi
if [ -n "$MODEL_LOCATION" ]; then
    sed -i "s|\./GoogleNews-vectors-negative300.bin|${MODEL_LOCATION}|" Config.py
else
    MODEL_LOCATION="./model_file.bin"
fi
if [ -n "$MODEL_URL" ]; then
    echo "Downloading pre-trained model..."
    wget ${MODEL_URL} -o ${MODEL_LOCATION}
    sed -i "s|\./GoogleNews-vectors-negative300.bin|${MODEL_LOCATION}|" Config.py
fi

PATH="$PATH:/home/newuser/.local/lib/python3.7/"
python3 Http.py
