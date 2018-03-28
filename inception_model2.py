
# coding: utf-8

# In[ ]:


import numpy as np
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
import h5py 

import os
import sys
import traceback
import shutil

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework import graph_util

from scipy.misc import imread, imsave, imresize
import scipy.ndimage
import matplotlib
matplotlib.use('TkAgg') # choose appropriate rendering backend
from matplotlib import pyplot as plt



K.set_learning_phase(0)

model = applications.InceptionV3(include_top=True, weights='imagenet',input_shape=(299, 299, 3))

print model.summary()

SERVING="serve"
export_path = 'INCEPTION3'
HEIGHT=299
WIDTH=299
#config = model.get_config()
#weights = model.get_weights()
#new_model = Sequential.from_config(config)
#new_model.set_weights(weights)

sess = K.get_session()
g = sess.graph
g_def = graph_util.convert_variables_to_constants(sess, 
                      g.as_graph_def(),
                      [model.output.name.replace(':0','')])

with tf.Graph().as_default() as g_input:
    input_b64 = tf.placeholder(shape=(1,),
                               dtype=tf.string,
                               name='b64')
    tf.logging.info('input b64 {}'.format(input_b64))

    image = tf.image.decode_image(input_b64[0])#input_bytes)
    image_f = tf.image.convert_image_dtype(image, dtype=tf.float16)
    input_image = tf.expand_dims(image_f, 0)
    input_data = tf.image.resize_bilinear(input_image, [HEIGHT, WIDTH], align_corners=False)
    #input_data = preprocess_image(image_r)
    output = tf.identity(input_data, name='input_image')




# Convert to GraphDef
g_input_def = g_input.as_graph_def()


with tf.Graph().as_default() as g_combined:
    x = tf.placeholder(tf.string, name="b64")

    im, = tf.import_graph_def(g_input_def,
                              input_map={'b64:0': x},
                              return_elements=["input_image:0"])

    pred, = tf.import_graph_def(g_def,
             input_map={model.input.name: im},
             return_elements=[model.output.name])

    with tf.Session() as session:
        inputs = {"image_bytes": tf.saved_model.utils.build_tensor_info(x)}
        outputs = {"output":tf.saved_model.utils.build_tensor_info(pred)}
        signature =tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )


        """Convert the Keras HDF5 model into TensorFlow SavedModel."""

        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess=session,
            tags=[tag_constants.SERVING],
            signature_def_map={ signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature },
        )
        builder.save()

