import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("ml/connect4_model_84accuracy", 'rb'))

def probability(board):
    X = board.T.flatten()
    return model.predict_proba([X])