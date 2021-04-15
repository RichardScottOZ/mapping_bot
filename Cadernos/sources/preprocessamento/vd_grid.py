import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import geopandas as gpd
# import pyproj as pyproj
import verde as vd
# from shapely import geometry
# import time
# from bokeh.plotting import figure, output_file, show
# from geopandas import GeoDataFrame
plt.rcParams['figure.dpi'] = 300


# ------------------------ Input raw data -------------------------------------
# -** Need to be inproved to a dictionary, organized database.
#     Maybe Fatiando a Terra's pooch library **
# ------------- Oppening the files delivered by the Brasilian Mineral Resources
#  Research Company formated as .XYZ, selecting columns, skiping metadata rows,
#

# -------------------- Setting Columns to work --------------------------------
# g14area
'''g14area_cols = 'ALTURA BARO COSMICO CTB CTCOR CTEXP DATA eTh eU FIDUCIAL GPSALT HORA KB KCOR KPERC LATITUDE LIVE_TIME LONGITUDE MDT TEMP THB THCOR THKRAZAO UB UCOR UKRAZAO UTHRAZAO UUP X UTME Y UTMN'.split(" ")
g14area_df = pd.read_csv('~/graphite_git/resources/xyz/Area_14_gama.XYZ',
                         names=g14area_cols,
                         delim_whitespace=True,
                         skiprows=8,
                         usecols=['UTME', 'UTMN', 'LONGITUDE', 'LATITUDE', 'X',
                                  'Y','MDT',
                                  'eTh', 'eU', 'KPERC', 'CTCOR',
                                  'THKRAZAO', 'UTHRAZAO', 'UKRAZAO'])
g14area_df.dropna(inplace=True)
g14area_df['geometry'] = [geometry.Point(x, y) for
                          x, y in zip(g14area_df['UTME'], g14area_df['UTMN'])]
g14area_df.to_csv('
                  ~/graphite_git/resources/csv/gama/g14area_df.csv',
                  index=False)
                  index=False)

# 1105
g1105_cols = 'KB DATA BARO UB THB COSMICO CTB UUP ALTURA KPERC eU eTh CTEXP UTHRAZAO UTME UTMN UKRAZAO MDT THKRAZAO LIVE_TIME CTCOR KCOR THCOR UCOR HORA GPSALT LATITUDE FIDUCIAL TEMP LONGITUDE'.split(" ")
g1105_df = pd.read_csv('~/graphite_git/resources/xyz/1105_GamaLine.XYZ',
                       names=g1105_cols,
                       delim_whitespace=True,
                       skiprows=11,
                       usecols=['UTME', 'UTMN', 'LONGITUDE', 'LATITUDE',
                                'MDT', 'eTh', 'eU', 'KPERC', 'CTCOR',
                                'THKRAZAO', 'UTHRAZAO', 'UKRAZAO'])
g1105_df.dropna(inplace=True)
g1105_df['geometry'] = [geometry.Point(x, y) for
                        x, y in zip(g1105_df['UTME'], g1105_df['UTMN'])]
g1105_df.to_csv('~/graphite_git/resources/csv/gama/g1105_df.csv', index=False)

# 1039
g1039_cols = 'UTME UTMN LONGITUDE LATITUDE MAGR eTh eU KPERC CTCOR MAGB MAGC MAGD THB UB KB CTB FIDU TEMP ALTE ALTB'.split(" ")
g1039_df = pd.read_csv('~/graphite_git/resources/xyz/spaulo_rjaneiro_sp.xyz',
                       names=g1039_cols,
                       delim_whitespace=True,
                       skiprows=5,
                       usecols=['UTME', 'UTMN', 'LONGITUDE', 'LATITUDE',
                                'CTCOR', 'KPERC', 'eU', 'eTh',
                                'MAGR'])
# There were a Type problem wtih UTME/UTMN column g1039_df,
#    so i needed to set df.astype('float32'), i dont remember where or when
#    this problem showed up.
g1039_df.dropna(inplace=True)
g1039_df['geometry'] = [geometry.Point(x, y) for
                        x, y in zip(g1039_df['UTME'].astype('float32'),
                                    g1039_df['UTMN'].astype('float32'))]
g1039_df['THKRAZAO'] = (g1039_df.eTh/g1039_df.KPERC)
g1039_df['UKRAZAO'] = (g1039_df.eU/g1039_df.KPERC)
g1039_df['UTHRAZAO'] = (g1039_df.eU/g1039_df.eTh)
g1039_df.to_csv('~/graphite_git/resources/csv/gama/g1039_df.csv', index=False)
'''
# ---------------------- Channel manipulation ---------------------------------
# CHANNEL RATIOS --------------------------------------------------------------
'''
g1039_df['THKRAZAO'] = (g1039_df.eTh/g1039_df.KPERC)
g1039_df['UKRAZAO'] = (g1039_df.eU/g1039_df.KPERC)
g1039_df['UTHRAZAO'] = (g1039_df.eU/g1039_df.eTh)
g1039_df.to_csv('~/graphite_git/resources/csv/gama/g1039_df.csv', index=False)
'''
# -------------------------- Concatenate surveys ------------------------------
'''frames = [g1039_df, g1105_df, g14area_df]
region = pd.concat(frames)
region.to_csv('~/graphite_git/resources/csv/gama/concat_region.csv',
              index=False)'''

# ------------------- Loading treated data from database-----------------------
'''
g1039_df = pd.read_csv('~/graphite_git/resources/csv/gama/g1039_df.csv')

g1105_df = pd.read_csv('~/graphite_git/resources/csv/gama/g1105_df.csv')
'''
g14area_df = pd.read_csv('~/graphite_git/resources/csv/gama/g14area_df.csv')
'''
region = pd.read_csv('~/graphite_git/resources/csv/gama/concat_region.csv')

'''
# ------------SETTING COORDS & Projection function ----------------------------
# --------- Configure Geometry and Coordinate Reference System ----------------
'''region['geometry'] = [geometry.Point(x, y)for
                      x, y in zip(region['UTME'], region['UTMN'])]'''
# projection = pyproj.Proj(proj="merc", lat_ts=dados.LATITUDE.mean())
# "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
# '+proj=utm +zone=23 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
'''
Proj('+proj=utm +zone=23 +south +ellps=WGS84', preserve_units=False)
projection = pyproj.Proj(proj="wgs", lat_ts=dados.LATITUDE.mean())
'''
# -----------------------  GRIDDING WITH SPLINES  -----------------------------

dados = g14area_df

# -------------------------- BLOCKED REDUCTIONS -------------------------------
# 1 - vd.BlockReduce---------------- CREATING A REDUCER  ----------------------
# Block reduction are dividing the region in blocks of a especified spacing

coordinates = (g14area_df['UTME'].values, g14area_df['UTMN'.values])
'''
reducer = vd.BlockReduce(np.median, spacing=2000)
b_coords, b_eU = reducer.filter(coordinates, dados.eU)
'''
# Figure 1 --------- Visualizing our decimated data sample spatial atribute
'''
plt.figure(figsize=(10, 10))
plt.scatter(b_coords[0], b_coords[1], c=b_eU, s=0.3)
plt.colorbar()
plt.title('Pontos coletados à distâncias médias de 2000 m')
plt.axis('scaled')
'''
'''
# 2 - ---- FITTING THE LINEAR MODEL OF DECIMATEDGRID WITH SPLINE  -------------
spline = vd.Spline()
# def spline.fit():
spline.fit(b_coords, b_eU)
'''

# 3 -------- PREDICTTING THE ACTUAL DATA WITH THE LINEAR MODEL ----------------
# the values of non-decimated dataset with the fitted linnear model
'''
predicted = spline.predict(coordinates)
'''
# Figure 2 ---------- Visualizing -------------------------------
'''
plt.figure(figsize=(7, 14))
plt.scatter(coordinates[0], coordinates[1], c=predicted, s=2)
plt.axis('scaled')
plt.colorbar()
plt.title("Valores preditos para posições amostradas")
'''
# 3.1 CALCULATING THE DIFERENCE BETWEEN PREDICTED AND SAMPLED DATA ------------
'''
residuals = dados.eU - predicted
'''
# Figure 3 -------- Visualizing -----------------------------------------
'''
scale = vd.maxabs(residuals)
plt.figure(figsize=(10, 10))
plt.scatter(coordinates[0], coordinates[1], c=residuals,
            s=2, cmap="RdBu_r", vmin=-scale, vmax=scale)
plt.colorbar()
plt.title("Diferença entre valores amostrados e preditos.")
plt.axis('scaled')
'''
# 4 ------- CREATING A SYNTHETIC SPACIAL GRID TO BE PREDICTED -----------------
# -- Generating a regular grid with VERDE by:
# Selecting region of the grid and the spacing between each pixel
'''
region = vd.get_region(coordinates)
# def vd.grid_coordinates():
grid_coords = vd.grid_coordinates(region, spacing=500)
'''
# 5 -------- PREDICTTING VALUES AT SYNTHETIC GRID WITH THE LINEAR MODEL -------
'''
# def splne.predict():
grid_eU = spline.predict(grid_coords)
'''
# Figure 4 ---------------Visualizing
'''
plt.figure()
plt.pcolormesh(grid_coords[0], grid_coords[1], grid_eU)
plt.axis('scaled')
plt.title(
    "From discrete 500 m irregular matrix to data to Grid of 500 m pixel")
plt.colorbar()
'''
# Built'n functions made to simplify data storage by creating a Xarray
'''
grid = spline.grid(spacing=500, data_names=['eU'])
'''
#  Figure 5 - Visualizing prediceted grid
'''
plt.figure()
grid.eU.plot()
plt.axis('scaled')
plt.title("Predito para o grid sintético de 500 m.")
grid.to_netcdf(
    '~/graphite_git/resources/tif/verde/area14/a14_05k_eU.nc')

'''

# ---------------------------- CHAINING OPERATIONS ----------------------------
# -------  Chanining configuration --------------------------------------------
'''
chain = vd.Chain([
    ('trend', vd.Trend(degree=6)),
    ('reduce', vd.BlockReduce(np.median, spacing=1500)),
    ('spline', vd.Spline()),
    ])
chain
'''

# -----------------------------------------------------------------------------
# fit
'''
chain.fit(coordinates, dados.eU)
'''
# Grid
'''
grid = chain.grid(spacing=500, data_names=['eU'], max_distance=500)
'''
# Visualizing
'''
plt.figure()
grid.eU.plot()
plt.axis('scaled')
'''

# ------------------------- Model Validation ----------------------------------
'''
chain.fit(*train)
chain.score(*test)

train, test = vd.train_test_split(coordinates, dados.eU, test_size=0.1,
                                  spacing=1000)
plt.figure()
plt.title('Blocked Train and Test data')
plt.plot(train[0][0], train[0][1], '.b', markersize=2)
plt.plot(test[0][0], test[0][1], '.r', markersize=2)

chain.fit(*train)
chain.score(*test)
'''
# K-Fold Cross Validation
'''
cv = vd.BlockKFold(spacing=1000, n_splits=10, shuffle=True)
scores = vd.cross_val_score(chain, coordinates, dados.eU, cv=cv)
'''

# --------------- AUTOMATIZING CHAINING OPERATIIONS ---------------------------
# Abrupt variations on the data values in small distances producess erros that
# can be overturned by increasing the degree of the polinomial fuction
#
# The counts of uranium radiation is strong in body of rocks
# leading the a strong signal in small areas and a high variance
# betwen the litologis that has uranium tha thoose that does not.


# ---------------------------- SELLECTING CHANNEL TO GRID ---------------------
[grid_parameters] = ['eU']


def chain(spacing, degree):
    print("chain_config begin")
    chain = vd.Chain([
        ('trend',  vd.Trend(degree=degree)),
        ('reduce', vd.BlockReduce(np.median, spacing=spacing)),
        ('spline', vd.Spline()),
        ])
    print(chain)
    return chain


def fit(row):
    print('#  Fitted')
    chain.fit(coordinates, dados[row])
    return fit


def grid(row, cell_size, max_distance, grid):
    grid = chain.grid(spacing=cell_size, data_names=[row])
    grid = vd.distance_mask(coordinates, maxdist=max_distance, grid=grid)
    grid[row].to_netcdf(row+'.nc')   ######### CREATE A ORGANIZED DATABASE
    grid[row].plot(figsize=(8, 8), cmap='magma')
    plt.axis('scaled')
    print('* '+row+' Fitted')


# --------------- Model Validation
sample_block_size = 100


def validation(row, test_size=0.1, spacing=sample_block_size):
    print("# Model validation begin")
    train, test = vd.train_test_split(coordinates, dados[row],
                                      test_size=test_size, spacing=spacing)
    chain.fit(*train)
    score = chain.score(*test)
    print(score)
    print("* Model validation end")


####  Cross-Validation
def cross_validation(row):
    print("# Cross Validation begin")
    cv = vd.BlockKFold(spacing=10000,n_splits=10,shuffle=True)
    scores = vd.cross_val_score(chain,coordinates,dados[row],cv=cv)

def process(tolerance=0.75):
    print("process begin")
    for row in grid_parameters:
        fit(row)
        validation(row)
        cross_validation(row)
        grid(row)


process()

Hilo, [19.03.21 16:30]
def process(tolerance=0.75):
    print("process begin")
    chain=chain(spacing=1000,degree=5)
    for row in grid_parameters:
        chain=fit(row)
        validation(row)
        cross_validation(row)
        grid(row)
!segundo tem que atribuit ao fit um valor de retorno
def process(tolerance=0.75):
    print("process begin")
    chain=chain(spacing=1000,degree=5)
    for row in grid_parameters:
        fit(row)
        validation(row)
        cross_validation(row)
        grid(row)