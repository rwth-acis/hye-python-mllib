#!/bin/bash
if [ -n "$SERVICE_PORT" ]; then
    sed -i "s/PRT = 8000/PRT = ${SERVICE_PORT}/" Http.py
fi
if [ -n "$MODEL_LOCATION" ]; then
    sed -i "s|MODEL_FILE = b\"./GoogleNews-vectors-negative300.bin\"|MODEL_FILE = b\"${MODEL_LOCATION}\"|" Word2Vec.py
else
    MODEL_LOCATION="./model_file.bin"
fi
if [ -n "$MODEL_URL" ]; then
    echo "Downloading pre-trained model..."
    wget ${MODEL_URL} -o ${MODEL_LOCATION}
    sed -i "s|MODEL_FILE = b\"./GoogleNews-vectors-negative300.bin\"|MODEL_FILE = b\"${MODEL_LOCATION}\"|" Word2Vec.py
fi

python3 Http.py
