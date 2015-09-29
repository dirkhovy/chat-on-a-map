from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import numpy as np
from skimage import color, measure
from skimage.measure import approximate_polygon
from skimage.morphology import binary_dilation, binary_erosion
from shapely.geometry import shape, Point, Polygon
w

# add layers
def get_contour(image):
    image_data = plt.imread(image)

    gimg = color.colorconv.rgb2grey(image_data)
    bwimg = gimg > 0
    bwimg = binary_dilation(bwimg, None)
    bwimg = ndimage.binary_fill_holes(bwimg)
    bwimg = binary_erosion(bwimg)
    contours = [approximate_polygon(new_s, 0.9) for new_s in measure.find_contours(bwimg, 0.5)]

    return bwimg, contours


def convert_to_lat_lng(xy, left_edge_lng, right_edge_lng, top_edge_lat, bottom_edge_lat, map_shape):
    print left_edge_lng, right_edge_lng, top_edge_lat, bottom_edge_lat
    geo = np.zeros_like(xy)
    dist_lng = (right_edge_lng - left_edge_lng)
    geo[:, 0] = left_edge_lng + (xy[:, 1] / map_shape[0]) * dist_lng
    dist_lat = (top_edge_lat - bottom_edge_lat)
    geo[:, 1] = top_edge_lat - (xy[:, 0] / map_shape[1]) * dist_lat

    return geo

def convert_to_map_size(xy, map_shape):
    print xy, map_shape
    geo = np.zeros_like(xy)
    geo[:, 1] = xy[:, 0] / xy[:, 0].max() * map_shape[0]
    geo[:, 0] = xy[:, 1] / xy[:, 1].max() * map_shape[1]

    return geo



coords = {'vollsmose':{'lllon': 10.386036, 'lllat': 55.400343, 'urlon': 10.457705, 'urlat': 55.419517},
          'bylderup':{'lllon': 8.86837, 'lllat': 54.889246, 'urlon': 9.445496, 'urlat': 55.065394}
}

bg_imgs = {'vollsmose': "/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/originals/Oversigtskort_Vollsmose_2.PNG",
           'bylderup': "/Users/dirkhovy/working/lowlands/sociolinguistics/chat-on-a-map/data/originals/Oversigtskort_Bylderup.PNG"
           # 'bylderup': '/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/bylderup_OSM.png',
           # 'vollsmose':'/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/vollsmose_OSM.png'
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



b2, im2 = get_contour('/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/annotations/V_AFA_FA.png')
b3, im3 = get_contour('/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/annotations/V_NKI_FA.png')


# display the whole thing

# m.imshow(plt.imread('/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/annotations/V_AFA_FA.png'),
#          interpolation='lanczos', origin='upper')
# m.imshow(im3, interpolation='lanczos', origin='upper')

# generate random points and place them on the map
x = np.random.uniform(coords[town]['lllon'], coords[town]['urlon'], 100) #longitudes
y = np.random.uniform(coords[town]['lllat'], coords[town]['urlat'], 100) #latitudes
x1, y1 = m((coords[town]['urlon']+coords[town]['lllon'])/2, (coords[town]['urlat']+coords[town]['lllat'])/2)
print x1, y1, (coords[town]['urlon']+coords[town]['lllon'])/2, (coords[town]['urlat']+coords[town]['lllat'])/2
map_width, map_height = m(coords[town]['urlon'], coords[town]['urlat'])
m.scatter(x1, y1, s=100)
m.scatter(100, 100, s=100)


print b2.shape
print background.shape

test = [np.array([[700, 0],
        [700, 100],
        [800,100],
        [800, 0]
])]
for c, contour in enumerate(test):

    # geo = convert_to_lat_lng(contour, coords[town]['lllon'], coords[town]['urlon'], coords[town]['urlat'], coords[town]['lllat'], background.shape)
    # geo = convert_to_map_size(contour, background.shape)
    x = contour[:, 0]#geo[:, 0]
    y = contour[:, 1]#geo[:, 1]
    print x[:10]
    print y[:10]
    print
    # x1, y1 = m(geo[:, 0], geo[:, 1])
    # print x1[:10]
    # print y1[:10]
    # print
    m.plot(x, y, linewidth=2)


plt.show()
