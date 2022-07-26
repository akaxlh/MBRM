# Two-path MATN

This is a multi-behavior recommender between MATN and GNMR. It models the user-side interactions and the item-side interactions jointly using MATN as encoder.

Please unzip the ML-10M data in Datasets/MultiInt-ML10M first.

Run labcode.py for Yelp and ML10M data, and use labcode_tmall.py for Online Retail data.

For example, to train and test MBRM model on Yelp data for like prediction, run
```
python labcode.py --mult 1e2 --reg 5e-3 --data yelp --target buy
```
