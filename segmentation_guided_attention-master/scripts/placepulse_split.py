import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = 'votes_clean.csv'
ATTRIBUTES = [
    'wealthy',
    'depressing',
    'safety',
    'lively',
    'boring',
    'beautiful',
    'all'
]
ROOT_PATH = 'votes'

if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    if not ROOT_PATH in os.listdir():
        os.mkdir(ROOT_PATH)
    for attr in ATTRIBUTES:
        print(attr)
        if not attr in os.listdir(ROOT_PATH):
            os.mkdir(f'{ROOT_PATH}/{attr}')
        attr_split = data[data['category'] == attr] if attr != 'all' else data
        train, val = train_test_split(attr_split, test_size=0.2, stratify=attr_split['category'])
        train.to_csv(f'{ROOT_PATH}/{attr}/train.csv', header=True, index=False)
        val.to_csv(f'{ROOT_PATH}/{attr}/val.csv', header=True, index=False)





