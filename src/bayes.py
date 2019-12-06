import util

## Constants
DATUM_WIDTH = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

## Module Classes

class Datum:
  """
  A datum is a pixel-level encoding of digits or face/non-face edge maps.

  Digits are from the MNIST dataset and face images are from the 
  easy-faces and background categories of the Caltech 101 dataset.
  
  
  Each digit is 28x28 pixels, and each face/non-face image is 60x74 
  pixels, each pixel can take the following values:
    0: no edge (blank)
    1: gray pixel (+) [used for digits only]
    2: edge [for face] or black pixel [for digit] (#)
    
  Pixel data is stored in the 2-dimensional array pixels, which
  maps to pixels on a plane according to standard euclidean axes
  with the first dimension denoting the horizontal and the second
  the vertical coordinate:
    
    28 # # # #      #  #
    27 # # # #      #  #
     .
     .
     .
     3 # # + #      #  #
     2 # # # #      #  #
     1 # # # #      #  #
     0 # # # #      #  #
       0 1 2 3 ... 27 28
   
  For example, the + in the above diagram is stored in pixels[2][3], or
  more generally pixels[column][row].
       
  The contents of the representation can be accessed directly
  via the getPixel and getPixels methods.
  """
  def __init__(self, data,width,height):
    """
    Create a new datum from file input (standard MNIST encoding).
    """
    DATUM_HEIGHT = height
    DATUM_WIDTH=width
    self.height = DATUM_HEIGHT
    self.width = DATUM_WIDTH
    if data == None:
      data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)]
    self.pixels = util.arrayInvert(convertToInteger(data))

  def getPixel(self, column, row):
    """
    Returns the value of the pixel at column, row as 0, or 1.
    """
    return self.pixels[column][row]

  def getPixels(self):
    """
    Returns all pixels as a list of lists.
    """
    return self.pixels

  def getAsciiString(self):
    """
    Renders the data item as an ascii image.
    """
    rows = []
    data = util.arrayInvert(self.pixels)
    for row in data:
      ascii = map(asciiGrayscaleConversionFunction, row)
      rows.append( "".join(ascii) )
    return "\n".join(rows)

  def __str__(self):
    return self.getAsciiString()



# Data processing, cleanup and display functions

def loadDataFile(filename, n,width,height):
  """
  Reads n data images from a file and returns a list of Datum objects.
  
  (Return less then n items if the end of file is encountered).
  """
  DATUM_WIDTH=width
  DATUM_HEIGHT=height
  fin = readlines(filename)
  fin.reverse()
  items = []
  for i in range(n):
    data = []
    for j in range(height):
      data.append(list(fin.pop()))
    if len(data[0]) < DATUM_WIDTH-1:
      # we encountered end of file...
      print "Truncating at %d examples (maximum)" % i
      break
    items.append(Datum(data,DATUM_WIDTH,DATUM_HEIGHT))
  return items

import zipfile
import os
def readlines(filename):
  "Opens a file or reads it from the zip archive data.zip"
  if(os.path.exists(filename)):
    return [l[:-1] for l in open(filename).readlines()]
  else:
    z = zipfile.ZipFile('data.zip')
    return z.read(filename).split('\n')

def loadLabelsFile(filename, n):
  """
  Reads n labels from a file and returns a list of integers.
  """
  fin = readlines(filename)
  labels = []
  for line in fin[:min(n, len(fin))]:
    if line == '':
        break
    labels.append(int(line))
  return labels

def asciiGrayscaleConversionFunction(value):
  """
  Helper function for display purposes.
  """
  if(value == 0):
    return ' '
  elif(value == 1):
    return '+'
  elif(value == 2):
    return '#'

def IntegerConversionFunction(character):
  """
  Helper function for file reading.
  """
  if(character == ' '):
    return 0
  elif(character == '+'):
    return 1
  elif(character == '#'):
    return 2

def convertToInteger(data):
  """
  Helper function for file reading.
  """
  if type(data) != type([]):
    return IntegerConversionFunction(data)
  else:
    return map(convertToInteger, data)

import copy
import random

# CONSTANTS
FACE_TRAINING = 451

def bayes_digit_training(percent):

  label_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
  n = (float(percent)/100) * 5000
  n = int(n)
  items = loadDataFile("/Users/aarti/ai_pa2/data/digitdata/trainingimages",5000,28,28)
  labels = loadLabelsFile("/Users/aarti/ai_pa2/data/digitdata/traininglabels",5000)

  data_indexes = []

  # digit: label_count = {0:val, 1:val, ..., 9:val}
  for i in range(n): # n = # times to randomly select a training image
    index = random.randint(0, (len(labels)-1))
    data_indexes.append(index)
    label_count[labels[index]] += 1

  label_prob = copy.deepcopy(label_count)

  for i in range(len(label_count)):
    temp = label_prob[i] / float(n)
    label_prob[i] = temp


  # initializing prob_map
  i = 0
  j = 0
  prob_map = {}
  for x in range(784):
    if j == 28:
      j = 0
      i += 1
    digit = 0
    while digit < 10:
      prob_map[(j, i, digit)] = 0
      digit += 1
    j += 1

  # loading prob map with feature occurence count for each training image
  counter = 0 # number of training images to process
  for x in range(n):
    i = 0
    j = 0
    for pixel in range(784):
      if j == 28:
        j = 0
        i += 1
      if items[data_indexes[counter]].getPixel(j, i) >= 1:
        prob_map[(j, i, labels[data_indexes[counter]])] += 1
      j += 1
    counter += 1

  # dividing each count by digit occurrence, stored in label_count
  i = 0
  j = 0
  for x in range(784):
    if j == 28:
      j = 0
      i += 1
    digit = 0
    while digit < 10:
      prob_map[(j, i, digit)] /= float(label_count[digit])
      digit += 1
    j += 1

  # print "training complete"
  return prob_map, label_prob

'''
  # checking that vals loaded properly -- will be deleted later
  i = 0
  j = 0
  for pixel in range(784):
    if j == 28:
      j = 0
      i += 1
    print "x : ", j, " | y : ", i, "count : ", prob_map[(j, i, 0)]
    j += 1
'''

def bayes_digit_testing(prob_map, label_prob_main):
  n = 1000
  items = loadDataFile("/Users/aarti/ai_pa2/data/digitdata/testimages",1000,28,28)
  labels = loadLabelsFile("/Users/aarti/ai_pa2/data/digitdata/testlabels",1000)

  counter = 0
  acc = 0
  for x in range(1000):
    label_prob = copy.deepcopy(label_prob_main)
    i = 0
    j = 0
    for x in range(784):
      # print "x : ", j, " | y : ", i
      if j == 28:
        j = 0
        i += 1
      digit = 0
      while digit < 10:
        if items[counter].getPixel(j, i) == 0:
          label_prob[digit] *= (1-prob_map[(j,i,digit)])
        else:
          label_prob[digit] *= prob_map[(j,i,digit)]
        digit += 1
      j += 1
      # print "at x : ", x, "| lP : ", label_prob

    max_element = max(label_prob, key=label_prob.get)
    if max_element == labels[counter]: acc += 1
    counter += 1

  acc = acc/float(1000)
  # print "testing complete"
  return acc

def bayes_digit_analysis():
  percentage = 10

  for x in range(10):
    i = 0
    print "acc at", percentage, "% training data : "
    while i < 10:
      prob_map, label_prob = bayes_digit_training(percentage)
      acc = bayes_digit_testing(prob_map, label_prob)
      print acc
      i += 1
    percentage += 10

def bayes_face_training(percent):

  label_count = {0:0, 1:0}
  n = (float(percent)/100) * 451
  n = int(n)
  items = loadDataFile("/Users/aarti/ai_pa2/data/facedata/facedatatrain",451,60,70)
  labels = loadLabelsFile("/Users/aarti/ai_pa2/data/facedata/facedatatrainlabels",451)

  data_indexes = []

  for i in range(n): # n = # times to randomly select a training image
    index = random.randint(0, (len(labels)-1))
    data_indexes.append(index)
    label_count[labels[index]] += 1

  label_prob = copy.deepcopy(label_count)

  for i in range(len(label_count)):
    temp = label_prob[i] / float(n)
    label_prob[i] = temp

  for x in range(2):
    temp = math.log(label_prob[x])
    label_prob[x] = temp

  # initializing prob_map
  i = 0
  j = 0
  prob_map = {}
  for x in range(4200):
    if j == 60:
      j = 0
      i += 1
    prob_map[(j, i, 0)] = 0
    prob_map[(j, i, 1)] = 0
    j += 1

  # loading prob map with feature occurence count for each training image
  counter = 0 # number of training images to process
  for x in range(n):
    i = 0
    j = 0
    for pixel in range(4200):
      if j == 60:
        j = 0
        i += 1
      if items[data_indexes[counter]].getPixel(j, i) >= 1:
        prob_map[(j, i, labels[data_indexes[counter]])] += 1
      j += 1
    counter += 1

  output_grid = smoothing(prob_map, n, label_count)
  return output_grid, label_prob

'''
  # checking that vals loaded properly -- will be deleted later
  i = 0
  j = 0
  for pixel in range(4200):
    if j == 60:
      j = 0
      i += 1
    print "x : ", j, " | y : ", i, "count : ", prob_map[(j, i, 1)]
    j += 1
'''

import math
def smoothing(input_grid, n, label_count):
  output_grid = {}
  zero_count = float(label_count[0])
  one_count = float(label_count[1])

  # load probabilities from input with k = .001
  i = 0
  j = 0
  for x in range(4200):
    if j == 60:
      j = 0
      i += 1
    output_grid[(j, i, 0)] = math.log(((float(input_grid[(j, i, 0)]) + .001) / (zero_count+.002)))
    output_grid[(j, i, 1)] = math.log(((float(input_grid[(j, i, 1)]) + .001) / (one_count+.002)))
    j += 1

  return output_grid

def bayes_face_testing(prob_map, label_prob_main):

  n = 150
  items = loadDataFile("/Users/aarti/ai_pa2/data/facedata/facedatatest",150,60,70)
  labels = loadLabelsFile("/Users/aarti/ai_pa2/data/facedata/facedatatestlabels",150)

  counter = 0
  acc = 0
  for x in range(150):
    label_prob = copy.deepcopy(label_prob_main)
    i = 0
    j = 0
    for x in range(4200):
      if j == 60:
        j = 0
        i += 1
      curr_label = 0
      while curr_label < 2:
        if items[counter].getPixel(j, i) == 0:
          val = prob_map[(j,i,curr_label)]
          label_prob[curr_label] += math.log((1 - math.exp(val)))
        else:
          label_prob[curr_label] += prob_map[(j,i,curr_label)]
        curr_label += 1
      j += 1

    max_element = max(label_prob, key=label_prob.get)
    if max_element == labels[counter]: acc += 1
    counter += 1

  acc = acc/float(150)
  return acc

def bayes_face_analysis():
  percentage = 10

  for x in range(10):
    i = 0
    print "acc at", percentage, "% training data : "
    while i < 10:
      prob_map, label_prob = bayes_face_training(percentage)
      acc = bayes_face_testing(prob_map, label_prob)
      print acc
      i += 1
    percentage += 10

def printGrid(grid, label):
  i = 0
  j = 0
  for pixel in range(4200):
    if j == 60:
      j = 0
      i += 1
    print "x : ", j, " | y : ", i, "count : ", grid[(j, i, label)]
    j += 1

# pick a number between 0-999
def bayes_digit_demo(image_index):
  prob_map, label_prob = bayes_digit_training(100)
  n = 1000
  items = loadDataFile("/Users/aarti/ai_pa2/data/digitdata/testimages",1000,28,28)
  labels = loadLabelsFile("/Users/aarti/ai_pa2/data/digitdata/testlabels",1000)

  i = 0
  j = 0
  for x in range(784):
    if j == 28:
      j = 0
      i += 1
    digit = 0
    while digit < 10:
      if items[image_index].getPixel(j, i) == 0:
        label_prob[digit] *= (1-prob_map[(j,i,digit)])
      else:
        label_prob[digit] *= prob_map[(j,i,digit)]
      digit += 1
    j += 1

  max_element = max(label_prob, key=label_prob.get)
  if max_element == labels[image_index]: print "correct prediction"
  else: print "incorrect prediction"
  print "predicted label : ", max_element, "| actual label : ", labels[image_index]
  print items[image_index]


def bayes_face_demo(image_index):
  prob_map, label_prob = bayes_face_training(100)
  n = 150
  items = loadDataFile("/Users/aarti/ai_pa2/data/facedata/facedatatest",150,60,70)
  labels = loadLabelsFile("/Users/aarti/ai_pa2/data/facedata/facedatatestlabels",150)

  i = 0
  j = 0
  for x in range(4200):
    if j == 60:
      j = 0
      i += 1
    curr_label = 0
    while curr_label < 2:
      if items[image_index].getPixel(j, i) == 0:
        val = prob_map[(j,i,curr_label)]
        label_prob[curr_label] += math.log((1 - math.exp(val)))
      else:
        label_prob[curr_label] += prob_map[(j,i,curr_label)]
      curr_label += 1
    j += 1

  max_element = max(label_prob, key=label_prob.get)
  if max_element == labels[image_index]: print "correct prediction"
  else: print "incorrect prediction"
  print "predicted label : ", max_element, "| actual label : ", labels[image_index]
  print items[image_index]  


# Testing
def _test():
  # bayes_digit_analysis()
  # bayes_face_analysis()

  ''' pick a number between 0-149 '''
  # bayes_face_demo(10)

  ''' pick a number between 0-999 '''
  # bayes_digit_demo(6)

if __name__ == "__main__":
  _test()

