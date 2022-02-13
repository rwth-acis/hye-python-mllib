from ctypes import *
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
punctuation = set({' ', '\n', '\t', '`', '~', '!', '@', '#', '$', '%', '^',\
                '&', '*', '(', ')', '-', '_', '=', '+', '[', ']', ';', ':',\
                '\'', '"', '\\', '|', '<', '>', ',', '.', '/', '?'})

# Load C library for computing word vectors
lib = CDLL('./libwordcenter.so')
lib.compute_center.restype = POINTER(c_float)
lib.compute_center.argtypes = [c_char_p, c_uint]
lib.load_model.restype = c_int
lib.load_model.argtypes = [c_char_p]
lib.get_model.restype = POINTER(c_float)
lib.get_dictionary.restype = c_char_p
lib.get_dimensionality.restype = c_longlong
lib.get_dictionary_size.restype = c_longlong

# Location of model file
MODEL_FILE = b"./GoogleNews-vectors-negative300.bin"
# Same value as in 'word_center.h'
MAX_WORD_LENGTH = 50

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Combines the words in the given list into one string which can be used by the
# c library
#
# wordList -> List of words of which the center is to be computed
#
# Returns: One string where every word is padded to the maximum word size
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def padWords(wordList):
    resultString = ""
    for word in wordList:
        resultString += word
        for i in range(MAX_WORD_LENGTH - len(word)):
            resultString += '\0'
    return bytes(resultString, "utf-8")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copys the values of the given c array into a list
#
# cFloatArray -> C type float array
#
# Returns: Python list of float numbers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def cFloatArrayToList(cFloatArray):
    pyArray = list()
    for i in range(int(lib.get_dimensionality())):
        pyArray.append(cFloatArray[i])
    return pyArray

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Removes stop words and punctuation from given list of words
#
# wordList -> Python list of words
#
# Same words as in given list but without stop words or punctuation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def removeStopWords(wordList):
    resultWords = list()
    for word in wordList:
        if not word in stop_words and not word in punctuation:
            resultWords.append(word)
    return resultWords

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copmutes the center of the given list of words using word2vec representation
#
# words -> List of words
#
# Returns: Vector of float numbers representing the center of the given words
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def computeCenter(words):
    wordList = removeStopWords(words)
    return cFloatArrayToList(lib.compute_center(padWords(wordList),\
        len(wordList)))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Initializes the C library by loading the model from the local binary file
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def loadModel():
    lib.load_model(MODEL_FILE)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Frees the memory allocated to hold the model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def freeModel():
    lib.free_model()