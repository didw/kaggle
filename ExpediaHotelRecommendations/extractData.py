#!/usr/bin python
#-*- coding: utf-8 -*-
from __future__ import print_function
import argparse
from os.path import dirname, isfile, join as path_join
import pickle
import numpy as np
from numpy import array
import os
import sys
import h5py
import datetime

reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('--output_h5', default='data/ABto108_2nd.h5')
parser.add_argument('--val_frac', type=float, default=0.2)
parser.add_argument('--test_frac', type=float, default=0)
parser.add_argument('--num_feat', type=int, default=9)
parser.add_argument('--num_frame', type=int, default=100)
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--encoding', default='utf-8')
args = parser.parse_args()


fname = 'data/train.csv'



############################## SAMPLE DATA ###########################################################################

#                   0          1               2                      3                     4                   5                          6            7          8 
#           date_time, site_name, posa_continent, user_location_country, user_location_region, user_location_city, orig_destination_distance,     user_id, is_mobile,
# 2014-08-11 07:46:59,         2,              3,                    66,                  348,              48862,                 2234.2641,          12,         0,
#    201301 ~ 2014-12,    2 ~ 53,          0 ~ 4,               0 ~ 239,             0 ~ 1027,          0 ~ 56508,             0.0 ~ 12407.9, 1 ~ 1198785,       0~1,
#          9       10          11          12               13                 14           15                   16
# is_package, channel,    srch_ci,    srch_co, srch_adults_cnt, srch_children_cnt, srch_rm_cnt, srch_destination_id,
#          1,       9, 2014-08-27, 2014-08-31,               2,                 0,           1,                8250,
#      0 ~ 1,  0 ~ 10,   `12~2558,   `12~2558,           0 ~ 9,             0 ~ 9,       0 ~ 8,           0 ~ 65107,

#                       17          18       19               20             21            22             23
# srch_destination_type_id, is_booking,     cnt, hotel_continent, hotel_country, hotel_market, hotel_cluster
#                        1,          0,       3,               2,            50,          628,             1
#                    0 ~ 9,      0 ~ 1, 1 ~ 269,           0 ~ 6,         0 ~ 6,     0 ~ 2117,        0 ~ 99

########################################################################################################################################


############################ ANALYZE ###################################

data_min = [1e9 for x in range(24)]
data_max = [0 for x in range(24)]
data_min[0] = data_min[11] = data_min[12] = '9999-99-99'
print(data_min)
head = 1
with open(fname) as f:
  for line in f.readlines():
    break
    if head == 1:
      head = 0
      continue
    for i, word in enumerate(line.split(',')):
      if word == '':
        continue
      if i in [0, 11, 12]:
        data_min[i] = min(data_min[i], word)
        data_max[i] = max(data_max[i], word)
        continue
      if i == 6:
        data_min[i] = min(data_min[i], float(word))
        data_max[i] = max(data_max[i], float(word))
        continue

      data_min[i] = min(data_min[i], int(word))
      data_max[i] = max(data_max[i], int(word))

for i, _ in enumerate(data_min):
  break
  print(i, '\t', data_min[i], '\t', data_max[i])





def date2week(date):
  y, m, d = date.split('-')
  return datetime.date(int(y), int(m), int(d)).isocalendar()[1]

def one_hot(data, len):
  d = np.zeros(len)
  d[data] = 1
  return d

def one_hot_range(data, len, step):
  d = np.zeros(len)
  d[int(data/step)] = 1
  return d

destFile = 'data/destinations.csv'
infoDest = [[0.0 for j in range(149)] for i in range(64994)]
head = 1
with open(destFile) as f:
  for line in f.readlines():
    if head == 1:
      head = 0
      continue
    words = line.split(',')
    idx = 0
    for i, d in enumerate(words):
      if i == 0:
        idx = int(d)
        continue
      infoDest[idx][i-1] = float(d)
def info_dest(idx):
  return infoDest[idx]



########################## CONVERT DATA ###############################
data = []
label = []
head = 1
with open(fname) as f:
  for line in f.readlines():
    convData = []
    if head == 1:
      head = 0
      continue
    words = line.split(',')

    # pass clicked data
    if words[18] == 0:
      continue

    # pass incomplete data
    inCompleteData = 0
    for chk in words:
      if chk == '':
        inCompleteData = 1
    if inCompleteData == 1:
      continue

    print('conv data length: ', len(convData))
    convData.append(date2week(words[0].split(' ')[0]))    # date_time ( 2013-01-07 00:00:02 ~ 2014-12-31 23:59:59 )
    convData.append(one_hot(int(words[1]), 54))       # site name      ( 2 ~ 53 )
    convData.append(one_hot(int(words[2]), 5))        # posa continent ( 0 ~ 4 )
    convData.append(one_hot(int(words[3]), 240))      # user_location_country ( 0 ~ 239 )
    convData.append(one_hot(int(words[4]), 1028))     # user_location_region ( 0 ~ 1027 )
    convData.append(one_hot(int(words[5]), 56509))    # user_location_city ( 0 ~ 56508 )
    print('conv data length: ', len(convData))
    convData.append(one_hot_range(float(words[6]), 1000, 12408.0/999))     # orig_destination_distance(0.0 ~ 12407.9)
    #convData.append(one_hot(int(words[7]), 54))      # user_id ( 1 ~ 1198785 )
    convData.append(one_hot(int(words[8]), 2))        # is_mobile ( 0 ~ 1 )
    convData.append(one_hot(int(words[9]), 2))        # is_package ( 0 ~ 1 )
    convData.append(one_hot(int(words[10]), 11))      # channel ( 0 ~ 10 )
    convData.append(date2week(words[11]))             # srch_ci ( 2012-02-15 ~ 2558-03-15 )
    print('conv data length: ', len(convData))
    convData.append(date2week(words[12]))             # srch_co ( 2012-09-04 ~ 2558-03-16 )
    convData.append(one_hot(int(words[13]), 10))      # srch_adults_cnt ( 0 ~ 9 )
    convData.append(one_hot(int(words[14]), 10))      # srch_children_cnt ( 0 ~ 9 )
    convData.append(one_hot(int(words[15]), 9))       # srch_rm_cnt ( 0 ~ 8 )
    convData.append(one_hot(int(words[16]), 65108))   # srch_destination_id ( 0 ~ 65107 )
    convData.append(info_dest(int(words[16])))        # srch_destination_id ( 0 ~ 65107 )
    print('conv data length: ', len(convData))
    convData.append(one_hot(int(words[17]), 10))      # srch_destination_type_id ( 0 ~ 9 )
    convData.append(one_hot(int(words[18]), 2))       # is_booking ( 0 ~ 1 )
    convData.append(one_hot(int(words[19]), 270))     # cnt ( 1 ~ 269 )
    print('conv data length: ', len(convData))
    convData.append(one_hot(int(words[20]), 7))       # hotel_continent ( 0 ~ 6 )
    convData.append(one_hot(int(words[21]), 213))     # hotel_country ( 0 ~ 212 )
    convData.append(one_hot(int(words[22]), 2118))    # hotel_market ( 0 ~ 2117 )
    print('conv data length: ', len(convData))
    data.append(convData)
    print('data length: ', len(data))
    label.append(int(words[23]))

