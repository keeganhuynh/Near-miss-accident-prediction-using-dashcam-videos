from pykml import parser
import pandas as pd
import numpy as np
import math

def CoorReturn(str):
  x = ""
  y = ""
  z = ""
  flag1 = False
  flag2 = False
  for i in str:
    if (i == '/t'):
      continue;
    if (i == ',' and flag1==False):
      flag1 = True
      continue
    if (i == ',' and flag1==True):
      flag2 = True
      continue
    if (flag1 == False):
      x = x + i
    if (flag1 == True):
      y = y + i
    if (flag2 == True):
      z = z + i
  return float(x), float(y), float(z)

def CoorsReturn(str):
  coors=[]
  demo = ''
  flag = True
  if (str[0] == '\n'):
    flag = False
  for i in str:
    if (flag == False):
      flag == True
      continue;
    demo = demo + i
    if (i=='\n'):
      coors.append(CoorReturn(demo))
      demo = ''
  return coors

def process(KML_file_path):
  with open(KML_file_path) as fobj:
      folder = parser.parse(fobj).getroot().Document
  coors = []
  for pm in folder.Placemark:
    line = pm.LineString.coordinates
  
  j = 0
  for i in line.text:
    if (i == '\n'): 
      j = j + 1
      continue
    if (i == '\t'): 
      j = j + 1
      continue
    if (i == ' '): 
      j = j + 1
      continue
    break
  line = line.text[j:]
  coors = CoorsReturn(line)
  return coors

def norm2(coor2, coor1):
  lat1 = coor1[0]
  lat2 = coor2[0]
  lon1 = coor2[1]
  lon2 = coor2[1]
  R = 6371000
  f1 = lat1 * math.pi/180
  f2 = lat2 * math.pi/180
  delf = (lat2 - lat1) * math.pi/180
  delp = (lon2 - lon1) * math.pi/180
  a = math.sin(delf/2) * math.sin(delf/2) + math.cos(f1) * math.cos(f2) * (math.sin(delp/2) * math.sin(delp/2))
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  return R*c

def ms2kmh(v):
  return v*3.6

def VelocityExtract(vec_file, KML_file_path,fps):
  coors = process(KML_file_path)
  f = open(vec_file, 'w')
  for i in range(len(coors)-1):
    for j in range(fps):
      line = str(ms2kmh(norm2(coors[i+1],coors[i]))) + '\n'
      f.write(line)