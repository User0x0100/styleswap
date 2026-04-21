from enum import IntEnum


class SampleType(IntEnum):
    IMAGE = 0 #raw image

    FACE_BEGIN = 1
    FACE = 1                        #aligned face unsorted
    FACE_PERSON = 2                 #aligned face person
    FACE_TEMPORAL_SORTED = 3        #sorted by source filename
    FACE_END = 3

    QTY = 4