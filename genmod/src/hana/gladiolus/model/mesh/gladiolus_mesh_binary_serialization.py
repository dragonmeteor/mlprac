import os
from typing import List

TYPE_TAG = -1
VALUE_TAG = -3
POSITION_TAG = 1
TEX_COORD_TAG = 2
TRIANGLE_TAG = 3
TEXTURE_TAG = 4

TYPE_ID_GLADIOLUS_MESH = 90019
TYPE_ID_CACHED = 40022
TYPE_ID_CACHE_KEY = 40023
TYPE_ID_STRING = 40006
TYPE_ID_FILE_PATH = 40020


def binary_serialize_gladiolus_mesh(
        vertices: List[List[float]],
        tex_coords: List[List[float]],
        triangles: List[List[int]],
        png_file_name: str,
        output_file_name: str):
    relpath = os.path.relpath(png_file_name, os.path.dirname(output_file_name))
    return {
        TYPE_TAG: TYPE_ID_GLADIOLUS_MESH,
        VALUE_TAG: {
            POSITION_TAG: vertices,
            TEX_COORD_TAG: tex_coords,
            TRIANGLE_TAG: triangles,
            TEXTURE_TAG: {
                TYPE_TAG: TYPE_ID_CACHED,
                VALUE_TAG: {
                    TYPE_TAG: TYPE_ID_CACHE_KEY,
                    VALUE_TAG: [
                        "ImageTexture",
                        [
                            {TYPE_TAG: TYPE_ID_FILE_PATH, VALUE_TAG: [True, relpath]},
                            {TYPE_TAG: TYPE_ID_STRING, VALUE_TAG: "clamp"},
                            {TYPE_TAG: TYPE_ID_STRING, VALUE_TAG: "clamp"}
                        ]
                    ]
                }
            }
        }
    }
