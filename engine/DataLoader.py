import pandas as pd

def load_data():
    df = pd.read_csv('engine/NEES_Database__ACI_369_Rectangular_Columns.csv')
    """
    print(df.iloc[5,0])
    print(df.head())
    """

    x, y = df.iloc[2:, 5:-1], df.iloc[2:, -1]
    #print(x.iloc[0,0])
    return x, y