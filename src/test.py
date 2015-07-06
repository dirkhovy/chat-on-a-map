import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import numpy as np


coords = {'vollsmose':{'lllon': 10.386036, 'lllat': 55.400343, 'urlon': 10.457705, 'urlat': 55.419517},
          'bylderup':{'lllon': 8.86837, 'lllat': 54.889246, 'urlon': 9.445496, 'urlat': 55.065394}
}

bg_imgs = {#'vollsmose': "/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/originals/Oversigtskort_Vollsmose_2.PNG",
           #'bylderup': "/Users/dirkhovy/working/lowlands/sociolinguistics/chat-on-a-map/data/originals/Oversigtskort_Bylderup.PNG"
           'bylderup': '/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/bylderup_OSM.png',
           'vollsmose':'/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/vollsmose_OSM.png'
           }

town = 'vollsmose'

m = Basemap(llcrnrlon=coords[town]['lllon'],
            llcrnrlat=coords[town]['lllat'],
            urcrnrlon=coords[town]['urlon'],
            urcrnrlat=coords[town]['urlat'],
            lat_ts=2, resolution='l', projection='merc',
            lon_0=(coords[town]['urlon'] + coords[town]['lllon']) / 2.0,
            lat_0=(coords[town]['urlat'] + coords[town]['lllat']) / 2.0)

# load the appropriate background image
background = plt.imread(bg_imgs[town])
m.imshow(background, interpolation='lanczos', origin='upper')

# add layers
im2 = plt.imread('/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/annotations/V_NKI_FA.png')
im3 = plt.imread('/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/annotations/V_NKI_FP.png')

# display the whole thing

m.imshow(im2, interpolation='lanczos', origin='upper')
m.imshow(im3, interpolation='lanczos', origin='upper')

# generate random points and place them on the map
x = np.random.uniform(coords[town]['lllon'], coords[town]['urlon'], 100) #longitudes
y = np.random.uniform(coords[town]['lllat'], coords[town]['urlat'], 100) #latitudes
x1,y1 = m(x,y)
m.scatter(x1, y1, s=100, c='r', marker="o", cmap=cm.jet, alpha=0.4)

print m(200,200,inverse=True)

plt.show()
