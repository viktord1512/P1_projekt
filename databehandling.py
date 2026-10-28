import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

heart = pd.read_csv("heart.csv")

# hvor er der nuller (der er kun nuller på cholesterol og en på restingbp)
cols = ["Age", "RestingBP", "Cholesterol", "MaxHR"]
print((heart[cols] == 0).sum())
print()

# # ændrer vores 0ler til not a number så imputeren kan se hvor den skal impute
# heart["Cholesterol"] = heart["Cholesterol"].replace(0,np.nan)
#
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer()
#
#
# heart.to_csv("heart_behandlet.csv", index=False, header=True)
# print((heart["Cholesterol"] == 0).sum())
