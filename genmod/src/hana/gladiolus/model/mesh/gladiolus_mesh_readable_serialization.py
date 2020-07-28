import os
from typing import List


def readable_serializer_position(v):
    return {
        "func": "position",
        "type": "Point2d",
        "value": [float(v[0]), float(v[1])],
    }


def readable_serializer_tex_coord(t):
    return {
        "func": "texCoord",
        "type": "Vector2d",
        "value": [float(t[0]), float(t[1])],
    }


def readable_serialize_triangle(t):
    return {
        "func": "triangle",
        "type": "Point3i",
        "value": [int(t[0]), int(t[1]), int(t[2])],
    }


def readable_serialize_gladiolus_mesh(
        vertices: List[List[float]],
        tex_coords: List[List[float]],
        triangles: List[List[int]],
        png_file_name: str,
        output_file_name: str):
    children = []

    for vertex in vertices:
        children.append(readable_serializer_position(vertex))
    for tex_coord in tex_coords:
        children.append(readable_serializer_tex_coord(tex_coord))
    for tri in triangles:
        children.append(readable_serialize_triangle(tri))
    relpath = os.path.relpath(png_file_name, os.path.dirname(output_file_name))
    children.append({
        "func": "texture",
        "type": "Cached",
        "key": {
            "type": "CacheKey",
            "protocol": "ImageTexture",
            "parts": [
                {
                    "type": "FilePath",
                    "relative": True,
                    "value": relpath,
                },
                "clamp",
                "clamp",
            ]
        }
    })
    return {
        "type": "gladiolus.Mesh",
        "children": children,
    }
