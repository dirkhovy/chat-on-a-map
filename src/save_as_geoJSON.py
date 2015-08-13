import os
import argparse
import json
import math

from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, measure
from scipy import ndimage
from skimage.morphology import binary_dilation
from skimage.measure import approximate_polygon, regionprops
from skimage.segmentation import clear_border
import pandas as pd

parser = argparse.ArgumentParser(description="""read PNG files, detect polygons, and output as geoJSON objects""")
parser.add_argument("input", help="input file", nargs='+')
parser.add_argument("--output", help="output file name", required=False, default="output")
parser.add_argument('--property-file', help="Read additional informant properties from this file")
args = parser.parse_args()

coords = {'vollsmose':{'left_edge_lng': 10.386036,
                       'bottom_edge_lat': 55.400343,
                       'right_edge_lng': 10.457705,
                       'top_edge_lat': 55.419517,
                       'map_shape': (859, 1676)
                      },
          'bylderup':{'lllon': 8.86837, 'lllat': 54.889246, 'urlon': 9.445496, 'urlat': 55.065394,
                      'map_shape': (800, 1679)
                     }}

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, item)


def convert_to_lat_lng(xy, left_edge_lng, right_edge_lng, top_edge_lat, bottom_edge_lat, map_shape):
    """Return a transformed coordinate array with the same dimensions as `xy` in (lat, lng) order"""
    geo = np.zeros_like(xy)
    # Pixel coordinates for latitude are reversed
    dist_lat = (top_edge_lat - bottom_edge_lat)
    relative_dist_from_top = xy[:, 0] / map_shape[0]
    geo[:, 0] = bottom_edge_lat + (1 - relative_dist_from_top) * dist_lat

    dist_lng = (left_edge_lng - right_edge_lng)
    geo[:, 1] = left_edge_lng - (xy[:, 1] / map_shape[1]) * dist_lng
    return geo

def get_polygons(image, town):
    """
    extract the shapes and their contours
    :param image:
    :return:
    """
    image_data = plt.imread(image)

    gimg = color.colorconv.rgb2grey(image_data)
    bw_img = gimg > 0

    # make sure it's closed
    bw_img = binary_dilation(bw_img, None)
    bw_img = ndimage.binary_fill_holes(bw_img)

    bw_img = clear_border(bw_img)

    labeled_img = measure.label(bw_img, background=0)

    polygons = []

    for region in regionprops(labeled_img):
        # skip small images
        if region.area < 100:
            continue
        for contour in measure.find_contours(bw_img == region.label, 0.5):
            # Is shape closed?
            if tuple(contour[0]) == tuple(contour[-1]):
                poly = approximate_polygon(contour, tolerance=5)

                geo = convert_to_lat_lng(poly, **coords[town])
                #             y = np.abs(y - map_height)

                lat = geo[:, 0]
                lng = geo[:, 1]
                polygons.append(Polygon(zip(lng, lat)))


    return polygons


# Properties
subject_props = None
if args.property_file:
    subject_props = pd.read_csv(args.property_file, sep=';')
    subject_props = subject_props.set_index('alias')



# Join individual files in single output file
output_file = open(args.output, 'w')
outputs = {"type": "FeatureCollection",
           "features": []}

for file_name in sorted(args.input):
    town_initial, subject_id, map_type = os.path.basename(file_name).replace('.png', "").split('_')
    town = 'byllerup' if town_initial == 'B' else 'vollsmose'

    polygons = get_polygons(file_name, town)

    multi_polygon = {"type": "MultiPolygon", "coordinates": []}

    for poly in polygons:
        multi_polygon['coordinates'].append(mapping(poly)['coordinates'])

    # save geoJSON
    geojson = {"type": "Feature",
               "geometry": multi_polygon,
               "properties": {"town": town, "subject_id": subject_id, "map_type": map_type}
               }

    if subject_props is not None:
        props = subject_props.ix[subject_id].to_dict()
        # Omit all undefined values
        for k in list(props.keys()):
            if isinstance(props[k], float) and math.isnan(props[k]):
                del props[k]
        # geojson['properties']['gender'] = props['gender']
        geojson['properties'].update(props)

    outputs['features'].append(geojson)

output_file.write(json.dumps(outputs, cls=NumpyAwareJSONEncoder))