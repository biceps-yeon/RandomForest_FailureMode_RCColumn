import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('engine/NEES_Database__ACI_369_Rectangular_Columns.csv')
    """
    print(df.iloc[5,0])
    print(df.head())
    """

    #y=Failure Mode:1:Flexural, 2: Flexure-Shear, 3: Shear
    spec_num, x, y = df.iloc[2:321, 0].astype(int), df.iloc[2:321, 5:-1], df.iloc[2:321, -1]
    #print(x.iloc[2:321, 0])
    #print(spec_num)

    #Load header name
    """['Section depth (h) [in.]', 'Section width (b) [in.]', 'd1 [in.]', 'a [in.]', 'a/d1', 
       'Longi. bars along first face (perp.)', 'Bar dia. [in.]',
       'Longi. bars along second face (perp.)', 'Bar dia. [in.].1',       
       'Longi. bars in middle layers(perp.)', 'Bar dia. [in.].2',        
       'fy (longi. reinf.) [psi]', 'pL (longi. reinf.)',
       'Trans. reinf. legs perp. to load',
       'Trans. bar dia. [in.]', 'Spacing of trans. reinf. (s) [in.]',     
       'fy (trans. reinf.) [psi]', 'pt (trans. reinf. volumetric ratio)', 
       'pv (trans. reinf. ratio)', 's/d1 (primary)', 
       'Seismic hoops', 'f'c [psi]', 'Axial load(P) [kips]',
       'Axial load ratio', 'Test configuration']
    """
    usd_head=[0,1,2, 6,7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 33, 43]

    # Excluded specimens at my paper
    excld_spec=[12,13,14,15,16,19,31,40,44,76,77,78,79,80,81,82,83,106,148,151,159,192,193,194,195,
      196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,224,225,226,230,231,232,233,234,
      235,236,237,286,295,296,297,298,299,300,301,302,303,304,305,306,307,314,315,316,317,318,319]
    excld_row = np.where(spec_num.isin(excld_spec))[0].tolist()
    sel_row = np.where(~spec_num.isin(excld_spec))[0].tolist()
    #print(sel_row)

    x_sel=x.iloc[sel_row, usd_head]
    y_sel=y.iloc[sel_row]
    #print(y_sel)

    return x_sel, y_sel
