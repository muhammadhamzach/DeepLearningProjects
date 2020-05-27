import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import argparse
import os
import numpy as np
import pandas as pd
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from yolo_utils import read_classes, read_anchors
from yad2k.models.keras_yolo import yolo_head
from yolo_eval import yolo_eval
from predict import predict
import h5py

sess = tf.compat.v1.Session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)    
yolo_model = load_model('./model_data/yolo.h5')

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

out_scores, out_boxes, out_classes = predict(sess, yolo_model, scores, boxes, classes, class_names, "giraffe.jpg")