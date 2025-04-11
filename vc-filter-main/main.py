# Apr-10-2024
# main.py

import cv2 as cv
from pathlib import Path
from time import perf_counter

from utils_directory import init_directory
import vc_filter


# Pick one of the image file
# -------------------------------------
image = 'face 2048x2048.jpg'
# image = 'Puerto Rico.jpg'
# image = 'fan.png'
# image = 'shapes.png'
# image = 'test_1.png'
# image = 'test_2.png'
# -------------------------------------


def main():

    init_directory('images_out')

    filename = Path(image)
    name = filename.stem
    extension = filename.suffix

    path_in = str(Path.cwd() / 'images_in' / str(name + extension))
    image_in = cv.imread(path_in, cv.IMREAD_UNCHANGED)

    # ---------------------------------------------------------
    start_time = perf_counter()

    image_edges = vc_filter.apply(image_in, 2.7)

    time_spent = perf_counter() - start_time
    # ---------------------------------------------------------

    path_edges = str(Path.cwd() / 'images_out' / str(name + '_edges' + extension))
    cv.imwrite(str(path_edges), image_edges)

    print(f'\nTime spent: {time_spent:.3f} sec')


if __name__ == "__main__":
    main()
