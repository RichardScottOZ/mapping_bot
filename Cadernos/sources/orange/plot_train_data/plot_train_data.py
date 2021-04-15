# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd

plt.rcParams['figure.dpi'] = 200


df = pd.read_csv(
    '~/graphite_git/sources/orange/plot_train_data/fccnd_train_data.csv')

'''
# Scatter of litol
x = df['X']
y = df['Y']

# Get unique names of species
uniq = list(set(df['Litologia']))

# Set the color map to match the number of species
z = range(1,len(uniq))
hot = plt.get_cmap('nipy_spectral')
cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

# Plot each species
for i in range(len(uniq)):
    indx = df['Litologia'] == uniq[i]
    plt.scatter(x[indx], y[indx], s=5,
                color=scalarMap.to_rgba(i),label=uniq[i])

plt.xlabel('X - UTME')
plt.ylabel('Y - UTMN')
plt.title('Train sample data')
plt.legend(loc='upper left')
plt.show()


'''

# Use the 'hue' argument to provide a factor variable
sns.lmplot( x=df['X'], y=df['Y'], data=df['Litologia'], fit_reg=False, hue='Litologia', legend=False)

# Move the legend to an empty part of the plot
plt.legend(loc='lower right')

plt.show()

#data = pd.read_csv('~/graphite_git/sources/orange/plot_train_data/fccnd_train_data.csv')

