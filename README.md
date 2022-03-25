# HyE - Python MlLib
This repository contains the code of a small HTTP server implemented in Python used to compute Matrix Factorization and word2vec word embeddings.
The part of the How's your Experience project and, more specifically, called by the [HyE YouTube Recommendations service](https://github.com/rwth-acis/hye-youtube-recommendations/) in order to generate the machine learning models it uses to evaluate the similarity of users' YouTube ratings.

## Setup
This service was developed using Python version 3.8 and pip version 20.0.2.
Furthermore, one of its core dependencies, the [SPARK MLlib](https://spark.apache.org/docs/latest/api/python/index.html), requires a Java version below Java 17.
If these requirements are met, install the dependencies by running `pip install -r required.txt` and start the server with `python Http.py`.

### Word2vec model
The service additionally depends on a pre-trained word2vec model with a specific structure.
This model has to be provided as a binary text file which starts with a [long long int](https://www.programiz.com/c-programming/c-data-types) denoting the number of words contained within that model and a second long long int denoting the dimensionality of the respective word2vec vectors.
Following that, the model file features byte arrays of length 50 to hold the words, each of which is followed by a list of float values representing the respective word vector.
One such pre-trained models featuring 3 million words with 300 dimensions can be downloaded [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

## Config
The server tries to connect to port 8000 by default.
This can be adjusted in the Config.py file.
The name and location of the word2vec model file can also be adjusted there.

## Development
The service additionally relies on a C library for the word2vec computations.
The library code is given in the `word_center.c` and `word_center.h` files and compiled to a library object with the following command.
```
cc -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -fPIC -shared -o libwordcenter.so word_center.c
```
This requires the following packages:
`gcc libc-dev`

## Service paths
The Python server implements two paths, one for the matrix factorization and one for the word2vec embeddings available under `/matrix-factorization` and `/word2vec` respectively.

### Matrix factorization
The matrix factorization implementation creates and stores models under a name, which is provided as a path parameter i.e.,: `/matrix-factorization/<MODEL_NAME>`.
Such a model is generated with a POST request featuring user rating data in the payload formatted as JSON with the following structure:
```
{'ratings': {user_u: {item_i: rating_ui}}, 'rank': rank, 'iterations': iterations, 'lambda': lambda}
```
Where `user_u`, `item_i`, and `rating_ui` are integers, as well as `rank` and `iterations`, and `lambda` is a double.
For further information, please refer to the [official documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.recommendation.ALS.html?highlight=matrix%20factorization#pyspark.mllib.recommendation.ALS.train)

The model can then be retrieved with a GET to that same URI which returns a JSON object of the following format:
```
{'userFeatures': {user_u: [vector_u1, ..., vector_un], item_i}, 'productFeatures': {item_i: [vector_i1, ..., vector_in], item_i}}
```
Where `n` denotes the rank of the factorized feature matrix.

Models can also be deleted by sending a DELETE to `/matrix-factorization/<MODEL_NAME>`

### Word2vec
The word2vec implementation works slightly differently.
Through a GET request, the word2vec model is loaded into memory.
The service then replies to POST requests featuring JSON arrays of strings as payloads by computing the center of these words according to the loaded word2vec model and returns it as a JSON array of double values.
Once the word2vec model is no longer needed, the memory should be freed again via a DELETE request.
