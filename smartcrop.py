import numpy as np
import cv2
import sys
import getopt
from matplotlib import pyplot as plt


def detectEdges(image):
  edges = cv2.Canny(image,300,300)
  plt.subplot(121),plt.imshow(image,cmap = 'gray')
  plt.title('Original Image'), plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(edges,cmap = 'gray')
  plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
  # edgeMap = plt * 255
  edges = cv2.GaussianBlur(edges,(5,5),0)
  edges = cv2.GaussianBlur(edges,(5,5),0)
  cv2.imwrite("fah.jpg",edges)

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
  img = cv2.imread(inputFile)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30),
      flags = cv2.cv.CV_HAAR_SCALE_IMAGE
  )
  detectEdges(img)
  print "Found {0} faces!".format(len(faces))
  for (x,y,w,h) in faces:
    print "sup"
    cv2.rectangle(img,(x,y),(x+w,y+h),(25,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


  # crop positioning
  imHeight,imWidth = img.shape[:2]
  print imWidth
  middle = (int(imWidth) / 2) - (int(width) / 2)
  cv2.rectangle(img, (20,20), (40,40), (0,255,0), 2)
  cv2.imwrite(outputFile,img)

if __name__ == "__main__":
   main(sys.argv[1:])