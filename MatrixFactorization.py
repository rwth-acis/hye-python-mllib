from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import traceback

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Turns the provided data in dictionary form into a Spark compatible RDD
#
# sc            -> The Spark Context from which the function is executed
# ratingsDict   -> user-item ratings as a dictionary ({user_id: {item, rating}})
# Returns: The ratings dictionary as Spark RDD or None on error
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def dictToRDD(sc, ratingsDict):
    try:
        # Turn dictionary into list of Spark Ratings
        ratingList = list()
        for user in ratingsDict:
            userRatings = ratingsDict[user]
            for item in userRatings:
                ratingList.append(Rating(user, item, userRatings[item]))
        # Create Ratings RDD from list
        return sc.parallelize(ratingList)
    except:
            traceback.print_exc()
            return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trains a Matrix Factorization model using Alternating Least Squares on the
# provided rating data using the given parameters
#
# sc            -> The Spark Context from which the function is executed
# ratings       -> user-item ratings as a dictionary ({user_id: {item, rating}})
# rank          -> Number of latent user/item features
# iterations    -> Number of iterations performed by ALS
# lambdaVal     -> A regularization factor
#
# Returns: The trained model as a MatrixFactorizationModel object
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def trainModel(sc, ratings, rank, iterations, lambdaVal):
    try:
        return ALS.train(dictToRDD(sc, ratings), rank, iterations, lambdaVal)
    except:
        traceback.print_exc()
        return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Evaluates the given model on the provided test data
#
# model     -> The model to evaluate
# testData  -> The test data as dictionary ({user_id: [{item, rating}]})
#
# Returns: The Mean Squared Error of the predicted ratings compared to the
# actual ratings
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def evaluateModel(model, testData):
    try:
        seSum = 0
        seCount = 0
        for user in testData:
            ratings = testData[user]
            for item in ratings:
                prediction = model.predict(user, item)
                actual = ratings[item]
                seSum = (actual - prediction) ** 2
                seCount += 1
        # rdd = dictToRDD(sc, testData)
        # print("Test data:")
        # print(rdd.collect())
        # se = rdd.map(lambda rating:\
        #     (rating[2] - model.predict(rating[0], rating[1])) ** 2)
        # print("Squared errors:")
        # print(se.collect())
        return seSum / seCount
    except:
        traceback.print_exc()
        return None
