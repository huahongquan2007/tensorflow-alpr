import collections
import itertools
import math
import sys
import os

import cv2
import numpy
import tensorflow as tf

import common
import model


def letter_probs_to_code(letter_probs):
    output = "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))
    return output.replace("_", "")


def predict(list_image_path, param_values):
    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()

    # Execute the model at each scale.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        for image_name in list_image_path:
            im_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE) / 255.
            im_gray = cv2.resize(im_gray, (128, 64))

            print("-------------")
            feed_dict = {x: numpy.stack([im_gray])}
            feed_dict.update(dict(zip(params, param_values)))
            y_val = sess.run(y, feed_dict=feed_dict)

            letter_probs = (y_val[0,
                            0,
                            0, 1:].reshape(
                10, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)

            present_prob = common.sigmoid(y_val[0, 0, 0, 0])
            print("input", image_name)
            print("output", letter_probs_to_code(letter_probs))

if __name__ == "__main__":

    f = numpy.load(sys.argv[2])
    param_values = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
    list_images = ["{}/{}".format(sys.argv[1], file_name) for file_name in os.listdir(sys.argv[1])]
    predict(list_image_path=list_images, param_values=param_values)


