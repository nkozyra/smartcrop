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

def findUnique(image):
  # blocks = 
  print image[0]

def main(argv):
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
  eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
  inputFile = ''
  outputFile = ''
  width = 300
  height = 300
  opts, args = getopt.getopt(argv,"hi:o:w:ht:",["ifile=","ofile="])
  for opt, arg in opts:
    if opt == "-i":
      inputFile = arg
    if opt == "-o":
      outputFile = arg
    if opt == "-ht":
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
  findUnique(img)
  maxFaceX = 0
  maxFaceCenter = 0
  maxFaceRight = 0
  print "Found {0} faces!".format(len(faces))

  # crop positioning
  imHeight,imWidth = img.shape[:2]
  cropBias = int(imWidth)
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(25,0,255),2)
    cropBias = cropBias - x
    if (x+w) > maxFaceX:
      maxFaceX = x + w
      maxFaceCenter = (x+w) - (w/2)
      maxFaceRight = (x+w)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


  cropBiasPerc = (cropBias * 1.0) / imWidth

  middleX = (int(imWidth) / 2) - (int(width) / 2)
  middleY = (int(imHeight) / 2) - (int(height) / 2)
  if maxFaceX != 0:
    middleX = maxFaceCenter - (int(width) / 2)
    if len(faces) > 1:
      middleX = int( middleX - (middleX * cropBiasPerc) )
    if middleX + (int(width)/2) > maxFaceRight:
      middleX = maxFaceRight - (int(width)/2)
  cv2.rectangle(img, (middleX,middleY), (middleX+int(width),middleY+int(height)), (0,255,0), 2)
  cv2.imwrite(outputFile,img)

if __name__ == "__main__":
   main(sys.argv[1:])