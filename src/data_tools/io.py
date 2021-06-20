import pandas as pd
import pathlib
import re
import math
import itertools as it
import numpy as np


DIMENSIONS = {
    "Circle": ["center_x", "center_y", "radius"],
    "Rectangle": ["center_x", "center_y", "grid_width", "grid_height"]
}


def _apply_pixelation(df, pixelation, dims):
    if pixelation == 1:
        return df, dims
    dims_ = {
        k: v /
        pixelation for k,
        v in dims.items() if not isinstance(
            v,
            str)}
    dims.update(dims_)
    df["x_px"] = df["x_px"] / pixelation
    df["y_px"] = df["y_px"] / pixelation
    df["x_px"] = df["x_px"].apply(lambda x: math.floor(x))
    df["y_px"] = df["y_px"].apply(lambda x: math.floor(x))
    return df, dims


def _discretize(df):
    df.x_px = df.x_px.astype(int)
    df.y_px = df.y_px.astype(int)
    return df


def _header_line(file_name):
    # -- end of yaml config --
    with open(file_name) as dat:
        for i, line in enumerate(dat):
            if line.find("# -- end of yaml config --") != -1:
                return i + 1  # where the data starts
        raise EOFError("End of yaml config not found in data.")


def _get_dimensions(file_name):
    print("getting the dimensions")
    dims = {dim: 0 for dim in ["shape"]}
    with open(file_name) as dat:
        for line in dat:
            if all(list(dims.values())):
                break
            if shapes := list(
                filter(
                    lambda shape: re.search(
                        shape, line), list(
                    DIMENSIONS.keys()))):
                dims["shape"] = shapes[0]
                dims.update({dim: 0 for dim in DIMENSIONS[shapes[0]]})
            for dim in dims.keys():
                if line.find(dim) != -1:
                    dims[dim] = float(re.search(r'\d+', line).group(0))
    return dims


def _transform(df, dims):

    x_transform, y_transform = get_center_transform(dims)

    dims['center_x'] -= x_transform
    dims['center_y'] -= y_transform

    if df is not None:
        df["x_px"] -= x_transform
        df["y_px"] -= y_transform
        df["x_px"] = df["x_px"].apply(lambda x: math.floor(x))
        df["y_px"] = df["y_px"].apply(lambda x: math.floor(x))

    return df, dims


def get_center_transform(dims=None, file_name=None):
    if not dims:
        dims = _get_dimensions(file_name)
    print(dims)
    if dims["shape"] == "Rectangle":
        return (dims['center_x'] - dims['grid_width']), (dims['center_y'] - dims['grid_height'])
    elif dims["shape"] == "Circle":
        return (dims['center_x'] - dims['radius']), (dims['center_y'] - dims['radius'])
    else:
        raise ValueError(f'{dims["shape"]} not implemented')


class CSVReader:
    def __init__(self, data_path, pixelation=1):
        self.data_path = data_path
        self.files = [p for p in pathlib.Path(self.data_path).glob("*.csv")]
        self.pixelation = pixelation

    def dimensions(self, file_name):
        dims = _get_dimensions(file_name)
        _, dims = _transform(None, dims)
        _, dims = _apply_pixelation(pd.DataFrame(
            {"x_px": [0], "y_px": [0]}), self.pixelation, dims)
        return dims

    @staticmethod
    def raw_data(file_name):
        return pd.read_csv(file_name, header=_header_line(file_name))

    def _single_file(self, file_name):
        dims = _get_dimensions(file_name)
        df = pd.read_csv(file_name, header=_header_line(file_name))
        print(dims)
        print(df)
        df, dims = _transform(df, dims)
        print(dims)
        print(df)
        #exit()
        df, dims = _apply_pixelation(df, self.pixelation, dims)
        df = _discretize(df)
        for dim in dims:
            df[dim] = dims[dim]
        return df

    def test_phase(self, file_name):
        df = self._single_file(file_name)

        led_on = df.loc[(df['led_1'] != 0)]
        led_off = df.iloc[:min(led_on.index), :]
        test_phase = df.iloc[max(led_on.index) + 1:, :]

        return test_phase

    @staticmethod
    def trajectory(df):
        # TODO what to do with all the non-movements due to pixelation?
        # a) remove all that are not stagnation in the original data
        # b) calculate average number of frames to change position, and
        # remove all those that are below that
        return list(zip(df["x_px"].values, df["y_px"]))


class Arena:
    def __init__(self, center_x, center_y, width, height):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height

    def transform(self):
        self.center_x = self.width / 2.0
        self.center_y = self.height / 2.0


class CircleArena(Arena):
    def __init__(self, center_x, center_y, radius):
        super().__init__(center_x, center_y, radius*2, radius*2)
    
    @property
    def radius(self):
        return self.width/2

    def build(self):
        x1 = np.arange(
            self.center_x,
            self.center_x +
            self.radius +
            1,
            dtype=int)
        y1 = np.arange(
            self.center_y,
            self.center_y +
            self.radius +
            1,
            dtype=int)
        x2 = np.arange(
            self.center_x -
            self.radius -
            1,
            self.center_x,
            dtype=int)
        y2 = y1
        x3 = x2
        y3 = np.arange(
            self.center_y -
            self.radius -
            1,
            self.center_y,
            dtype=int)
        x4 = x1
        y4 = y3

        x_1, y_1 = np.where((x1[:, np.newaxis] - self.center_x)
                            ** 2 + (y1 - self.center_y)**2 < self.radius**2)
        x_2, y_2 = np.where((x2[:, np.newaxis] + 1 - self.center_x)
                            ** 2 + (y2 - self.center_y)**2 < self.radius**2)
        x_3, y_3 = np.where((x3[:, np.newaxis] + 1 - self.center_x)
                            ** 2 + (y3 + 1 - self.center_y)**2 < self.radius**2)
        x_4, y_4 = np.where((x4[:, np.newaxis] - self.center_x)
                            ** 2 + (y4 + 1 - self.center_y)**2 < self.radius**2)

        first_quad = set(zip(x1[x_1], y1[y_1]))
        sec_quad = set(zip(x2[x_2], y2[y_2]))
        third_quad = set(zip(x3[x_3], y3[y_3]))
        fourth_quad = set(zip(x4[x_4], y4[y_4]))

        return first_quad | sec_quad | third_quad | fourth_quad


class RectangleArena(Arena):
    def __init__(self, center_x, center_y, grid_width, grid_height):
        super().__init__(center_x, center_y, grid_width, grid_height)

    def build(self):
        for state in it.product(range(int(math.ceil(self.width))), range(
                int(math.ceil(self.height)))):
            yield int(state[0] + (self.center_x - self.width / 2)), int(state[1] + (self.center_y - self.height / 2))


def Factory(shape, **dims):
    shapes = {
        "Circle": CircleArena,
        "Rectangle": RectangleArena
    }
    return shapes[shape](**dims)
