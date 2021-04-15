import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely import geometry
from geopandas import GeoDataFrame as gpd

import verde as vd
import time
plt.rcParams['figure.dpi'] = 300

'''
### Input raw data
 #Setting Columns to work

# Área 14       ALTURA BARO COSMICO CTB CTCOR CTEXP DATA eTh eU FIDUCIAL GPSALT HORA KB KCOR KPERC LATITUDE LIVE_TIME LONGITUDE MDT TEMP THB THCOR THKRAZAO UB UCOR UKRAZAO UTHRAZAO UUP X X_WGS Y Y_WGS
g14area_cols = 'ALTURA BARO COSMICO CTB CTCOR CTEXP DATA eTh eU FIDUCIAL GPSALT HORA KB KCOR KPERC LATITUDE LIVE_TIME LONGITUDE MDT TEMP THB THCOR THKRAZAO UB UCOR UKRAZAO UTHRAZAO UUP X UTME Y UTMN'.split(" ")
g14area_df = pd.read_csv('~/graphite_git/resources/xyz/Area_14_gama.XYZ',
                        names=g14area_cols,
                        delim_whitespace=True,
                        skiprows=8,
                        usecols=['UTME','UTMN','LONGITUDE','LATITUDE',
                                 'MDT','eTh','eU','KPERC','CTCOR','THKRAZAO','UTHRAZAO','UKRAZAO'])
g14area_df.dropna(inplace=True)
g14area_df['geometry'] = [geometry.Point(x, y) for x, y in zip(g14area_df['UTME'], g14area_df['UTMN'])]
g14area_df.to_csv('~/graphite_git/resources/csv/gama/g14area_df.csv',index=False)

# 1105
g1105_cols = 'KB DATA BARO UB THB COSMICO CTB UUP ALTURA KPERC eU eTh CTEXP UTHRAZAO UTME UTMN UKRAZAO MDT THKRAZAO LIVE_TIME CTCOR KCOR THCOR UCOR HORA GPSALT LATITUDE FIDUCIAL TEMP LONGITUDE'.split(" ")
g1105_df = pd.read_csv('~/graphite_git/resources/xyz/1105_GamaLine.XYZ',
                        names=g1105_cols,
                        delim_whitespace=True,
                        skiprows=11,
                        usecols=['UTME','UTMN','LONGITUDE','LATITUDE',
                                 'MDT','eTh','eU','KPERC','CTCOR','THKRAZAO','UTHRAZAO','UKRAZAO'])
g1105_df.dropna(inplace=True)
g1105_df['geometry'] = [geometry.Point(x, y) for x, y in zip(g1105_df['UTME'], g1105_df['UTMN'])]
g1105_df.to_csv('~/graphite_git/resources/csv/gama/g1105_df.csv',index=False)


# 1039
g1039_cols = 'UTME UTMN LONGITUDE LATITUDE MAGR THC UC KC CTCOR MAGB MAGC MAGD THB UB KB CTB FIDU TEMP ALTE ALTB'.split(" ")
g1039_df = pd.read_csv('~/graphite_git/resources/xyz/spaulo_rjaneiro_sp.xyz',
                        names=g1039_cols,
                        delim_whitespace=True,
                        skiprows=5,
                        usecols=['UTME','UTMN','LONGITUDE','LATITUDE',
                                 'CTCOR','KC','UC','THC','MAGR'])
# There were a Type problem wtih UTME/UTMN column g1039_df, so i needed to set df.astype('float32').
g1039_df.dropna(inplace=True)
g1039_df['geometry'] = [geometry.Point(x, y) for x, y in zip(g1039_df['UTME'].astype('float32'), g1039_df['UTMN'].astype('float32'))]
g1039_df.to_csv('~/graphite_git/resources/csv/gama/g1039_df.csv',index=False)


### Concatenate surveys
frames = [g1105_df]
region = pd.concat(frames)


####  bounds
#caconde     = 328909.5350343678146601,344583.3834759797900915,7614012.4019015226513147,7629149.9828700609505177
##caconde    = 329409,7614512,344084,7628650
##area14     = 292000,500000,7550000,7640000
##g1039      = 190000,400000,7375000,7700000
##regional   =
##           = 290000,350000,7600000,7700000
##Area_N_1105= 292819.2586019417503849,7599950.0203590430319309,344451.6613980582915246,7637672.3696409575641155

study_area = 292819.2586019417503849,344451.6613980582915246,7599950.0203590430319309,7637672.3696409575641155
dados = region[vd.inside((region.UTME, region.UTMN),region= study_area)]


#### ---------------------------  Configure Geometry and Coordinate Reference System -------------------------------------------
#dados['geometry'] = [geometry.Point(x, y) for x, y in zip(dados['UTME'].astype('float32'), dados['UTMN'].astype('float32'))]
#dados.to_csv('~/graphite_git/resources/csv/gama/cut/g_stdy_area.csv')

# '+proj=utm +zone=23 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
# "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
crs = '+proj=utm +zone=23 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
dados             = gpd(dados, geometry='geometry',crs=crs)

coordinates        = (dados.UTME, dados.UTMN)


#                             BLOCKED REDUCTIONS
# -reducer------------------- CREATING A REDUCER  -------------------------------------------
# Block reduction are dividing the region in blocks of a especified spacing
reducer = vd.BlockReduce(np.median, spacing= 960)
b_coords, b_UC = reducer.filter(coordinates, dados.UC)
print(b_coords,b_UC,b_UC.shape, dados.shape)

   ##### Visualizing our decimated data sample spatial atribute
plt.figure(figsize=(10,10))
plt.scatter(b_coords[0],b_coords[1], c= b_UC, s=2, )
plt.colorbar()
plt.title("Pontos coletados à distâncias mínimas de 1000 m")
plt.axis('scaled')

def reducer():
    vd.BlockReduce(np,median,spacing)








# --------------------------------------------------------------------------------------------

#### Channel manipulation

   #Cleaning negative values

dados = dados._get_numeric_data()
dados[dados.KPERC < 0 ] = 0
    #Adding channels
dados['THKRAZAO'] = dados['eTH'] / dados['KPERC']


### Channels to grid   ,
grid_parameters=['MDT']


### Counter
def timelapse(inicio):
    timelapse=str(time.process_time()-inicio)
    print("tempo decorrido:"+timelapse+"s")

####  Chanining configuration
def chain_config(spacing=500):
    inicio=time.process_time()
    print("chain_config begin")
    chain = vd.Chain([
		('trend',  vd.Trend(degree=2)),
		('reduce', vd.BlockReduce(np.median, spacing=spacing)),
		('spline', vd.Spline()),
		])
    print("chain_config end")
    timelapse(inicio)
    return chain


#### Then we fit within the chain
chain=chain_config()
cell_size=100
def fitting(row=1,max_distance=500):
	inicio=time.process_time()
	print('# '+row+' fitting begin')
	chain.fit(coordinates, dados[row])
	grid = chain.grid(spacing=cell_size, data_names=[row])
	grid = vd.distance_mask(coordinates, maxdist=max_distance, grid=grid)
	grid[row].to_netcdf('~/graphite_git/resources/tif/verde/cut/caconde/caconde_'+row+'.nc')  ## DataBase
	grid[row].plot(figsize=(8,8), cmap='magma')
	plt.axis('scaled')
	print('* '+row+' Fitted')
	timelapse(inicio)


#### Model Validation
sample_block_size = 100
def validation(row,test_size=0.1,spacing=sample_block_size):
	inicio=time.process_time()
	print("# Model validation begin")
	train, test = vd.train_test_split(coordinates, dados[row], test_size=test_size, spacing=spacing)
	chain.fit(*train)
	score=chain.score(*test)
	print(score) #treino ? teste? #verde
	print("* Model validation end")
	timelapse(inicio)
	return score


####  Cross-Validation
def cross_validation(row):
	inicio=time.process_time()
	print("# Cross Validation begin")
	cv = vd.BlockKFold(spacing=100,n_splits=10,shuffle=True)
	scores = vd.cross_val_score(chain,coordinates,dados[row],cv=cv)
	plt.figure()
	plt.hist(scores, bins ='auto')
	print(scores)
	print("* Cross Validation end")
	timelapse(inicio)


def process(tolerance=0.75):
    print("process begin")
    inicio=time.process_time()
    np.linspace(start=500, stop=1000, num=10)

    for row in grid_parameters:
        fitting(row)
        validation(row)
        cross_validation(row)

    timelapse(inicio)
	#inserir flag de tempo
	#if error >95% continue

process()

for params in other_parameters:
	np.arange(0,top+1,spacing)


