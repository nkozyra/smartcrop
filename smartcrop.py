import numpy as np
import cv2, getopt, sys
from matplotlib import pyplot as plt

def autocrop(image):
  pass


def main(argv):
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
  eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
  inputFile = ''
  outputFile = ''
  width = 300
  height = 300
  opts, args = getopt.getopt(argv,"hi:o:w:h:",["ifile=","ofile="])
  for opt, arg in opts:
    if opt == "-i":
      inputFile = arg
    if opt == "-o":
      outputFile = arg
    if opt == "-h":
      height = arg
    if opt == "-w":
      width = arg