import string
import numpy

DIGITS = string.digits
LETTERS = string.ascii_uppercase
CHARS = LETTERS + DIGITS + "_"
RANDOM_CHARS = LETTERS + DIGITS

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
  return 1. / (1. + numpy.exp(-a))

