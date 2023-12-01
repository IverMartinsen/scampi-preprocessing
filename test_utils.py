from pickle import TRUE
import unittest, os
import numpy as np
from openslide import deepzoom, OpenSlide
from skimage.measure import regionprops
from scampi_tools.utils import get_crop, get_region_props
from scampi_tools.utils_mac import tile_generator


class TestUtils(unittest.TestCase):
    def test_get_crop(self):

        test_size = 16, 16
        test_image = np.zeros(test_size, dtype=np.uint8)
        test_image[4:7, 4:7] = 255
        regionprop = regionprops(test_image)[0]

        crop1 = get_crop(regionprop, test_image, size=test_size)
        crop2 = get_crop(regionprop.bbox, test_image, size=test_size)

        self.assertTrue(np.all(crop1 == crop2))

    def test_get_tile(self):

        ###
        ### Replace with
        #
        # path_to_files = "/Users/ima029/Desktop/mrxs slides"
        # filenames = os.listdir(path_to_files)
        # path_to_slide = os.path.join(path_to_files, np.random.choice(filenames))

        path_to_slide = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Aktive prosjekter/SCAMPI/Microfossil data/6407_6-5 1670 mDC.mrxs"

        tile_size = 1024

        tile_maker = tile_generator(path_to_slide, tile_size)

        tile, level, address = next(tile_maker)

        self.assertTrue(
            np.all(
                tile
                == deepzoom.DeepZoomGenerator(
                    OpenSlide(path_to_slide),
                    tile_size=tile_size,
                    overlap=0,
                    limit_bounds=True,
                ).get_tile(level, address)
            )
        )


unittest.main()
