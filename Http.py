from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import traceback
import re
import ModelStorage as ms
import Word2Vec as w2v
import Config as config

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Helper function to build json response
#
# status        -> HTTP status code as int
# msg           -> Response payload
# content_type  -> Type of payload (default: text/plain)
#
# Returns: The given parameters as json object
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def build_response(status, msg, content_type = 'text\plain'):
    # TODO maybe validate args
    return {'status': status, 'msg': msg, 'content-type': content_type}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Parse the ratings data provided as json
#
# raw   -> Unparsed, unverified json data
#
# Returns: A dictionary of the form {user_u: {item_i: rating_ui}}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def parse_ratings(jsonObj):
    ratings = dict()
    try:
        if jsonObj == None or len(jsonObj) == 0:
            return ratings
        for user in jsonObj:
            rawRatings = jsonObj[user]
            userRatings = dict()
            for item in rawRatings:
                userRatings[int(item)] = float(rawRatings[item])
            ratings[int(user)] = userRatings
    except:
        traceback.print_exc()
        return 'Rating data invalid'
    return ratings

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Checks whether fields required to generate Matrix Factorization model are
# provided
#
# raw   -> Data required to build model as string
#
# Returns: The provided rating data and model parameters as dictionary
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def parse_model_data(raw):
    jsonObj = None
    try:
        jsonObj = json.loads(raw)
    except:
        traceback.print_exc()
        return 'Invalid json'
    required_args = ['ratings', 'rank', 'iterations', 'lambda']
    missing_args = list()
    for arg in required_args:
        if arg not in jsonObj:
            missing_args.append(arg)
    if len(missing_args) > 0:
        return 'Missing fields ' + str(missing_args)
    try:
        return {'ratings': parse_ratings(jsonObj['ratings']),\
            'rank': int(jsonObj['rank']),\
            'iterations': int(jsonObj['iterations']),\
            'lambda': float(jsonObj['lambda'])}
    except:
        traceback.print_exc()
        return 'Invalid data types'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Transforms given json word array string into list
#
# wordArray -> String in json array format
#
# Given json array as Python list
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def parse_word_array(wordArray):
    jsonArr = None
    try:
        jsonArr = json.loads(wordArray)
    except:
        traceback.print_exc()
        return 'Invalid json'
    return jsonArr

# TODO change main-page to something general
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Handles requests to the root path (/)
#
# method    -> HTTP method of request
# body      -> Request payload
#
# Returns: The resulting model built from provided data or error in case of
# invalid data
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main_page(method, body):
    if not method == "POST":
        return build_response(405,\
            'Method ' + method + ' not supported for this path')
    model_data = parse_model_data(body)
    if isinstance(model_data, str):
        return build_response(400, model_data)
    model_name = ms.updateModel(model_data)
    if model_name == None:
        return build_response(500, 'Error creating model')
    return build_response(200, model_name)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Handles requests to a path featuring a model name (/<model_name>)
#
# method    -> HTTP method of request
# path      -> Path information as list
# body      -> Request payload
#
# Returns: The respective model features as json
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def model(method, path, body):
    if len(path) < 1:
        return build_response(400, 'No model name provided')
    model_name = path[0]

    if method == "GET":
        model_features = ms.getFeatures(model_name)
        if model_features == None:
            return build_response(404, 'Model "' + model_name +\
                '" not found')
        return build_response(200, json.dumps(model_features), 'application/json')

    if method == "POST":
        model_data = parse_model_data(body)
        if isinstance(model_data, str):
            return build_response(400, model_data)
        model_features = ms.updateModel(model_data, model_name)
        if model_features == None:
            return build_response(500, 'Error updating model')
        return build_response(200, json.dumps(model_features), 'application/json')

    if method == "DELETE":
        if not ms.deleteModel(model_name):
            return build_response(500, 'Error deleting model')
        return build_response(200, 'Model ' + model_name + ' deleted')
    return build_response(405, 'Method ' + method +\
        ' not supported for this path')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Handles requests to the word2vec path
#
# method    -> HTTP method of request
# path      -> Path information as list
# body      -> Request payload
#
# Returns: The respective word2vec response as json
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def word2vec(method, path, body):
    if method == "GET":
        w2v.loadModel()
        return build_response(200, 'Loaded model', 'text\plain')

    if method == "POST":
        word_center = w2v.computeCenter(parse_word_array(body))
        if isinstance(word_center, str):
            return build_response(400, word_center)
        return build_response(200, json.dumps(word_center), 'application/json')

    if method == "DELETE":
        w2v.freeModel()
        return build_response(200, 'Model freed')
    return build_response(405, 'Method ' + method +\
        ' not supported for this path')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calls a function to handle the given request
#
# method    -> HTTP method of request
# path      -> Path information as string
# body      -> Request payload (default: None)
#
# Returns: The result of the respective request handler
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def route_request(method, path, body = None):
    path_split = path.split('/')
    if len(path_split) < 2 or path_split[1] == "":
        return main_page(method, body)
    elif path_split[1].lower() == "matrix-factorization":
        return model(method, path_split[2:], body)
    elif path_split[1].lower() == "word2vec":
        return word2vec(method, path_split[2:], body)
    return {"status": 404, "content-type": "text/plain", "msg":\
            "Unknown resource: " + path}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Helper function to determine whether given string can be cast to int
#
# string    -> Potentially numeric string
#
# Returns: True if string can be cast to int, False otherwise
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def isNumeric(string):
    return re.search('^[0-9]+$', string)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Helper function to parse headers of HTTP request
#
# rawHeaders    -> Request headers
#
# Returns: Parsed header values with uniform capitalization and missing
# required values replaced by default values
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def parse_headers(rawHeaders):
    # TODO maybe validate header values
    headers = dict()
    if not rawHeaders['Content-Length'] == None and\
        isNumeric(rawHeaders['Content-Length']):
            headers['content-length'] = int(rawHeaders['Content-Length'])
    elif not rawHeaders['content-length'] == None and\
        isNumeric(rawHeaders['content-length']):
            headers['content-length'] = int(rawHeaders['content-length'])
    else:
        headers['content-length'] = 0

    if not rawHeaders['Content-Encoding'] == None:
        headers['content-encoding'] = rawHeaders['Content-Encoding']
    elif not rawHeaders['content-encoding'] == None:
        headers['content-encoding'] = rawHeaders['content-encoding']
    else:
        headers['content-encoding'] = 'utf-8'
    return headers

# TODO Move main_page function to mf-function and make main_page function smth else
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# /
### POST -> Create new model from provided data and return name
# /matrix-factorization/<model_name>
### GET     -> Return model's feature vectors
### POST    -> Update model (create under provided name, if not taken)
### DELETE  -> Delete model
# /matrix-factorization/<model_name>/evaluate --- TODO ---
### GET     -> Return latest MSE
### POST    -> Compute and return MSE on provided data
# /word2vec
### GET     -> Loads the word2vec model (around 4GB) to memory
### POST    -> Compute the center of the given list of words in the vector space
### DELETE  -> Delete word2vec model from memory
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class CustomHandler(BaseHTTPRequestHandler):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Handles GET requests addressed to the server, calls the router with
    # the request parameters and sends the response returned by the invoked
    # function
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def do_GET(self):
        response = route_request('GET', self.path)
        self.send_response(response['status'])
        self.send_header('Content-Type', response['content-type'])
        self.end_headers()
        self.wfile.write(response['msg'].encode())

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Handles POST requests addressed to the server by parsing the requests
    # body, calling the router with the request parameters, and sending the
    # response returned by the invoked function
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def do_POST(self):
        body = None
        headers = parse_headers(self.headers)
        try:
            body = self.rfile.read(headers['content-length'])\
                .decode(headers['content-encoding'])
        except:
            traceback.print_exc()
            self.send_response(400)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write('Error getting request body'.encode())
            return
        response = route_request('POST', self.path, body)
        self.send_response(response['status'])
        self.send_header('Content-Type',response['content-type'])
        self.end_headers()
        self.wfile.write(response['msg'].encode())

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Handles DELETE requests addressed to the server, calls the router with
    # the request parameters and sends the response returned by the invoked
    # function
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def do_DELETE(self):
        response = route_request('DELETE', self.path)
        self.send_response(response['status'])
        self.send_header('Content-Type', response['content-type'])
        self.end_headers()
        self.wfile.write(response['msg'].encode())

# Start server
srv = HTTPServer(('',config.port), CustomHandler)
print('Server started on port %s' %config.port)
srv.serve_forever()
