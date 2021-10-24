import numpy as np
import enum


def add_indent(str_, num_spaces):
    s = str_.split('\n')
    s = [(num_spaces * ' ') + line for line in s]
    return '\n'.join(s)


def num_parameters(layer):
    def prod(arr):
        cnt = 1
        for i in arr:
            cnt = cnt * i
        return cnt

    cnt = 0
    for p in layer.parameters():
        cnt += prod(p.size())
    return cnt


class Mode(enum.Enum):
    NONE = 0
    ONE_PATH_FIXED = 1
    ONE_PATH_RANDOM = 2
    TWO_PATHS = 3
    ALL_PATHS = 4

