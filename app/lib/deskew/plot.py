from typing import List
import numpy as np
import matplotlib.pyplot as plt


def display_hough(h: float, a: List[float], d: List[float]) -> None:  # pylint: disable=invalid-name
    print("PASOK SA PLOT")
    plt.imshow(
        np.log(1 + h),
        extent=[np.rad2deg(a[-1]), np.rad2deg(a[0]), d[-1], d[0]],
        cmap=plt.gray,
        aspect=1.0 / 90,
    )
    plt.show()