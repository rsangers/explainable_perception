
# coding: utf-8

import pandas as pd
import numpy as np
import os

#constants
DATA_DIR = '../votes/'

images = set()
for attribute in os.listdir(DATA_DIR):
    data = pd.read_csv(f'{DATA_DIR}/{attribute}/val.csv')
    images |= set(data['left_id']) | set(data['right_id'])

result = pd.DataFrame({'id':list(images)})
for attribute in os.listdir(DATA_DIR):
    result[attribute] = 0

result.to_csv('../rank.csv',index=False)

