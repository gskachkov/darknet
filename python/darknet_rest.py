import threading
threading.stack_size(4*80*1024)

from ctypes import *
import math
import random
import logging

from flask import Flask, render_template, request
from flask.json import JSONEncoder
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES

import numpy as np
from PIL import Image, ImageDraw

UPLOAD_FOLDER = '/darknet/uploads'
STATIC_IMAGES = '/darknet/data'
STATIC_IMAGES_RESIZED = '/darknet/data/resized'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

from flask import Flask
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOADED_PHOTOS_DEST'] = STATIC_IMAGES

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class MyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return JSONEncoder.default(self, obj)


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

net = load_net("/darknet/cfg/yolov3-laptop.cfg", "yolov3-laptop_1300.weights", 0)
meta = load_meta("/darknet/cfg/laptop.data")

@app.route('/', methods=['GET'])
def index():
    logging.warning("Default")
    return 'Flask Dockerized'

@app.route('/prices', methods=['GET'])
def get_prices_methods():
    prices = {'ipad':450,'smartphone':250,'camera':350, 'iphone':300, 'macbook':800, 'notebook':500, 'headphones':100 }
    return jsonify(prices)

@app.route('/detect', methods=['POST'])
def detect_method():
    logging.warning("Start: try to detect")
    # r = detect(net, meta, "/darknet/data/dog.jpg")
    # print r
    print('detect method')
    if request.method == 'POST' and 'photo' in request.files:
        logging.warning("Do detect")
        file = request.files['photo']
        logging.warning("Do detect")
        # filename = photos.save(request.files['photo'])
        filename = file.filename
        filePath = STATIC_IMAGES + '/' + filename
        file.save(filename);
        logging.warning("Save file")

        logging.warning(filename)
        filePathResized = STATIC_IMAGES_RESIZED + '/' + filename

        logging.warning(filePath)
        img = Image.open(file)

        new_width  = 415
        new_height = new_width * img.height / img.width
        rect = { 'img_width': img.width, 'img_heigth': img.height, 'detect_width': new_width, 'detect_height': new_height }

        img_resized = img.resize(size=(new_width, new_height))
        img_resized.save(filePathResized, 'JPEG')
        img.close()
        img_resized.close()
        logging.warning(filePathResized)
        result = detect(net, meta, filePathResized)
        logging.warning('result')
        logging.warning(result)
        json_result = { 'sizes': rect, 'objects': result}
        return jsonify(json_result)
    return 'Wrong params'

if __name__ == '__main__':
     logging.warning("Start service!")
     app.run(host="0.0.0.0", port=8080)