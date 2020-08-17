# MBRM

Please unzip the ML-10M data in Datasets/MultiInt-ML10M first.

Run labcode.py for Yelp and ML10M data, and use labcode_tmall.py for Online Retail data.

For example:
```
python labcode.py --mult 1e2 --reg 5e-3 --data yelp --target buy
```
for training and testing on Yelp-Buy data.
