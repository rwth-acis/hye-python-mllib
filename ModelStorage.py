from pyspark import SparkContext
from pyspark.mllib.recommendation import MatrixFactorizationModel
import random
import time
import traceback
import shutil
import MatrixFactorization as mf

# Static Spark Context
sc = SparkContext(appName="HyeMatrixFactorization")
# Seed RNG with current time
random.seed(time.time())
# Characters used in generated model names
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Attempts to load the model stored under the given name
#
# name  -> Name of a previously stored model
#
# Returns: The retrieved model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def getModel(name):
    try:
        # Remove slashes to prevent accessing files on local system outside
        # model storage directory
        return MatrixFactorizationModel.load(sc, 'models/' + name.split('/')[-1])
    except:
        traceback.print_exc()
        return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Generates a random name for a model
#
# Returns: The retrieved model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def generateModelName():
    name = ""
    for i in range(32):
        name += CHARSET[int(random.random() * len(CHARSET))]
    return name

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Attempts to create a Matrix Factorization model from the provided data
#
# modelData -> Expects {'ratings': {user: {item: rating}}, 'rank': int,
# 'iterations': int, 'modelData': float} (see MatrixFactorization.train(...) for
# further information)
#
# Returns: The model trained based on the given ratings and parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def createModel(modelData):
    return mf.trainModel(sc, modelData['ratings'], modelData['rank'],\
            modelData['iterations'], modelData['lambda'])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Helper function to turn the feature vectors stored as RDDs to json format
#
# rdd -> Feature vectors in the form [(id, array('d', [values]))]
#
# Returns: Feature vectors in the form {id: [values]}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def rddToDict(rdd):
    result = dict()
    for tuple in rdd:
        try:
            result[tuple[0]] = list(tuple[1])
        except Exception as e:
            traceback.print_exc()
            return None
    return result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Helper function to package model features in json object
#
# model -> The Matrix Factorization model whose features we are interested in
#
# Returns: The provided model's feature vectors
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def getModelFeatures(model):
    userFeatures = rddToDict(model.userFeatures().collect())
    if userFeatures == None:
        return None
    productFeatures = rddToDict(model.productFeatures().collect())
    if productFeatures == None:
        return None
    return {'userFeatures': userFeatures, 'productFeatures': productFeatures};

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Deletes the given model with the provided name
#
# modelName -> Name of a previously stored Matrix Factorization model
#
# Returns: True if deletion was successful, False otherwise
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def deleteModel(modelName):
    try:
        shutil.rmtree('models/' + modelName, ignore_errors = False)
    except:
        traceback.print_exc()
        return False
    return True

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Stores the given model under the provided path
#
# model -> Matrix Factorization model
# path  -> Path on local file system where model should be stored
#
# Returns: True if storing was successful, False otherwise
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def saveModel(model, path):
    shutil.rmtree(path, ignore_errors = True)
    try:
        model.save(sc, path)
    except:
        traceback.print_exc()
        return False
    return True

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Attempts to load generate a Matrix Factorization model and stores it under the
# provided name on the local file system or a randomly generated name if none
# was provided
#
# modelData -> Data required to create the model
# modelName -> Name under which the model is supposed to be stored
#
# Returns: The feature vectors of the generated model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def updateModel(modelData, modelName = None):
    model = createModel(modelData)
    if model == None:
        return None
    if modelName == None or modelName == "":
        # New model
        modelName = generateModelName()
        if not saveModel(model, 'models/' + modelName):
            return None
        return modelName
    if not saveModel(model, 'models/' + modelName):
        return None
    return getModelFeatures(model)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Retrieves the model features for the provided model name
#
# modelName -> Name of a previously stored model
#
# Returns: The feature vectors of this model as json
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def getFeatures(modelName):
    model = getModel(modelName)
    if model == None:
        return None
    return getModelFeatures(model);
