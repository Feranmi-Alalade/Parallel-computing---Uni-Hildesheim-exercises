import pandas as pd

df = pd.read_csv(r"C:\Users\hp\Downloads\kaggle.csv", delimiter=';', encoding='ISO-8859-1')

print(df.columns)