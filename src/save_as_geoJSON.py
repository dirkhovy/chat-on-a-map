import glob
import os
from shapely.geometry import Polygon, mapping
from shapely.ops import transform
from functools import partial
import pyproj
from mpl_toolkits.basemap import Basemap
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, measure
from scipy import ndimage
from skimage.morphology import binary_erosion, binary_dilation
from skimage.measure import approximate_polygon
import argparse
import json

# parse command line options
parser = argparse.ArgumentParser(description="""read a ONG file and output as geoJSON object""")
parser.add_argument("input", help="input file", nargs='+')
parser.add_argument("--town", help="which town to choose", choices=['vollsmose', 'bylderup'], required=True)
parser.add_argument("--output", help="output file name", required=False, default="output")
args = parser.parse_args()


coords = {'vollsmose':{'left_edge_lng': 10.386036,
                       'bottom_edge_lat': 55.400343,
                       'right_edge_lng': 10.457705,
                       'top_edge_lat': 55.419517},
          'bylderup':{'lllon': 8.86837, 'lllat': 54.889246, 'urlon': 9.445496, 'urlat': 55.065394}
}

bg_imgs = {'vollsmose': "/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/originals/Oversigtskort_Vollsmose_2.PNG",
           'bylderup': "/Users/dirkhovy/working/lowlands/sociolinguistics/chat-on-a-map/data/originals/Oversigtskort_Bylderup.PNG"
#            'bylderup': '/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/bylderup_OSM.png',
#            'vollsmose':'/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/vollsmose_OSM.png'
           }


def get_contour(image):
    """
    extract the shapes and their contours
    :param image:
    :return:
    """
    image_data = plt.imread(image)

    gimg = color.colorconv.rgb2grey(image_data)
    bwimg = gimg > 0

    # make sure it's closed
    bwimg = binary_dilation(bwimg, None)
    bwimg = ndimage.binary_fill_holes(bwimg)
    # shrink it
    bwimg = binary_erosion(bwimg)

    contours = [approximate_polygon(new_s, 0.9) for new_s in measure.find_contours(bwimg, 0.5)]

    return np.where(bwimg > 0, 1, 0), contours


def magnify(org, x_factor, y_factor):
    """
    scale up to map size
    :param org:
    :param x_factor:
    :param y_factor:
    :return:
    """
    x, y = org.shape
    x1 = x * x_factor
    y1 = y * y_factor

    out = np.zeros((x1, y1))
    non_zeros = org.nonzero()
    for a, b in zip(non_zeros[0], non_zeros[1]):
        out[a*x_factor, b*y_factor] = org[a,b]

    return out


town = args.town

# fig, ax = plt.subplots(figsize=(15,15))

m = Basemap(llcrnrlon=coords[town]['left_edge_lng'],
            llcrnrlat=coords[town]['bottom_edge_lat'],
            urcrnrlon=coords[town]['right_edge_lng'],
            urcrnrlat=coords[town]['top_edge_lat'],
            lat_ts=2, resolution='l', projection='merc',
            lon_0=(coords[town]['right_edge_lng'] + coords[town]['left_edge_lng']) / 2.0,
            lat_0=(coords[town]['top_edge_lat'] + coords[town]['bottom_edge_lat']) / 2.0)

# load the appropriate background image
background = plt.imread(bg_imgs[town])
m.imshow(background, interpolation='lanczos', origin='upper')
map_width, map_height = m(coords[town]['right_edge_lng'], coords[town]['top_edge_lat'])
x_factor = map_height / background.shape[0]
y_factor = map_width / background.shape[1]
print background.shape
print "Map size: %d width, %d height" % (map_width, map_height)
print "Scaling factors: x=%.4f, y=%.4f" % (x_factor, y_factor)

show_contours = True

# projection calculator
project = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:4326'),
    pyproj.Proj('+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +no_defs'))

heatmap = np.zeros(background.shape[:2])

# for file_name in ['/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/annotations/V_AFA_FA.png',
#                 '/Users/dirkhovy/Dropbox/working/lowlands/sociolinguistics/chat-on-a-map/data/annotations/V_NKI_FA.png'
#                  ]:

output_file = open(args.output, 'w')
outputs = {"type": "FeatureCollection",
           "features": []
}

for file_name in sorted(args.input):

    town_initial, subject_id, map_type = os.path.basename(file_name).replace('.png', "").split('_')

    b2, im2 = get_contour(file_name)
    heatmap += b2

    if show_contours:
        multi_polygon = {"type": "MultiPolygon", "coordinates": []}

        for c, contour in enumerate(im2):
            # scale up dimensions to map size
            x, y = contour[:, 1] * x_factor, contour[:, 0] * y_factor

            # invert the y-axis
            y = np.abs(y - map_height)

            m.plot(x, y, linewidth=3)

            # convert to coordinates
            lng, lat = m(x, y, inverse=True)
            print lng.shape, lat.shape
            # cast as Polygon
            poly = Polygon(zip(lng, lat))
            #get center
            center = poly.centroid
            print center
            center_x, center_y = center.coords[0]
            xpt, ypt = m(center_x, center_y)

            poly_g = transform(project, poly)
            # compute area
            km2 = poly_g.area / 1000000.0
            plt.text(xpt, ypt-50, '%.3fW/%.3fN\n%.2fkm^2' % (center_x, center_y, km2))

            multi_polygon['coordinates'].append(mapping(poly)['coordinates'])

        # save geoJSON
        geojson = {"type": "Feature",
                   "geometry": multi_polygon,
                   "properties": {"town": args.town, "subject_id": subject_id, "map_type": map_type}
        }

        outputs['features'].append(geojson)

output_file.write(json.dumps(outputs))
# invert y
heatmap = np.flipud(heatmap)
# scale heatmap to map size
larger_heatmap = sp.ndimage.interpolation.zoom(heatmap, (x_factor, y_factor))
# plot
plt.imshow(larger_heatmap, alpha=0.3)
plt.show()
