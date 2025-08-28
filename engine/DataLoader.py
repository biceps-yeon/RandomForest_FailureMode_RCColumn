import pandas as pd

def load_data():
    df = pd.read_csv('engine/NEES_Database__ACI_369_Rectangular_Columns.csv')
    """
    print(df.iloc[5,0])
    print(df.head())
    """

    #y=Failure Mode:1:Flexural, 2: Flexure-Shear, 3: Shear
    spec_num, x, y = df.iloc[2:321, 0], df.iloc[2:321, 5:-1], df.iloc[2:321, -1]
    #print(x.iloc[0,0])
    
    #Load hear name
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
    x_sel=x.iloc[:, [0,1,2, 6,7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 33, 43]]
    #print(x_sel.columns)

    return x_sel, y