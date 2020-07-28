import json
import os

import cv2
import meshpy
import msgpack
import numpy
from scipy.spatial.qhull import ConvexHull

from hana.gladiolus.model.mesh.gladiolus_mesh_binary_serialization import binary_serialize_gladiolus_mesh
from hana.gladiolus.model.mesh.gladiolus_mesh_readable_serialization import readable_serialize_gladiolus_mesh


def generate_delaunay_mesh_file(
        png_file_name: str,
        output_file_name: str,
        maximum_area: float,
        dilation: int,
        show_steps: bool = False):
    """
    Generate a mesh that covers a given image.

    :param png_file_name: the input PNG file
    :param output_file_name: the name of the output file
    :param maximum_area: the maximum area of each triangle
    :param dilation: size of the kernel of the dilation operator
    """
    image = cv2.imdecode(numpy.fromfile(png_file_name, dtype=numpy.uint8), cv2.IMREAD_UNCHANGED)
    assert image.shape[2] == 4
    h, w = image.shape[0], image.shape[1]

    if output_file_name is None:
        input_dir = os.path.dirname(png_file_name)
        base_file_name = os.path.splitext(os.path.basename(png_file_name))[0]
        if input_dir == "":
            output_file_name = base_file_name + "_mesh.msgpck"
        else:
            output_file_name = input_dir + "/" + base_file_name + "_mesh.msgpck"
    output_name, output_extension = os.path.splitext(output_file_name)
    assert output_extension.lower() in ['msgpck', 'json']

    output_dir = os.path.dirname(output_file_name)
    output_base_file_name = os.path.splitext(os.path.basename(output_file_name))[0]
    if output_dir == "":
        output_prefix = output_base_file_name
    else:
        output_prefix = output_dir + "/" + output_base_file_name

    alpha_image = image[:, :, 3].reshape(h, w, 1)
    if show_steps:
        cv2.imwrite(output_prefix + "_alpha.png", alpha_image)

    dilation_kernel_size = dilation
    kernel = numpy.ones((dilation_kernel_size, dilation_kernel_size), dtype=numpy.uint8)
    dilated_image = cv2.dilate(alpha_image, kernel, iterations=1)
    if show_steps:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_prefix + "_dilated.png", dilated_image)

    dilated_non_zero_image = numpy.not_equal(dilated_image, 0).astype(numpy.uint8) * 255
    if show_steps:
        cv2.imwrite(output_prefix + "_dilated_non_zero.png", dilated_non_zero_image)

    nonzero_indices = numpy.nonzero(dilated_non_zero_image.reshape(h, w))
    nonzero_indices = numpy.stack(nonzero_indices, axis=1)
    convex_hull = ConvexHull(nonzero_indices)
    convex_hull_vertex_indices = convex_hull.vertices
    convex_hull_vertices = nonzero_indices[convex_hull_vertex_indices] + 0.5
    n = convex_hull_vertices.shape[0]
    if show_steps:
        convex_hull_image = numpy.zeros((h, w, 3), numpy.uint8)
        for i0 in range(n):
            i1 = (i0 + 1) % n
            p0 = convex_hull_vertices[i0]
            p1 = convex_hull_vertices[i1]
            cv2.line(convex_hull_image, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), (255, 255, 255))
        cv2.imwrite(output_prefix + "_convex_hull.png", convex_hull_image)

    max_area = maximum_area
    convex_hull_mesh_info = meshpy.triangle.MeshInfo()
    convex_hull_mesh_info.set_points(convex_hull_vertices.tolist())
    facets = [[i, (i + 1) % n] for i in range(n)]
    convex_hull_mesh_info.set_facets(facets)
    triangulation = meshpy.triangle.build(convex_hull_mesh_info, max_volume=max_area, volume_constraints=True)
    if show_steps:
        triangulation_image = numpy.zeros((h, w, 3), numpy.uint8)
        for tri in triangulation.elements:
            a, b, c = tri
            pa = triangulation.points[a]
            pb = triangulation.points[b]
            pc = triangulation.points[c]
            cv2.line(triangulation_image, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), (255, 255, 255))
            cv2.line(triangulation_image, (int(pb[1]), int(pb[0])), (int(pc[1]), int(pc[0])), (255, 255, 255))
            cv2.line(triangulation_image, (int(pc[1]), int(pc[0])), (int(pa[1]), int(pa[0])), (255, 255, 255))
        cv2.imwrite(output_prefix + "_triangulation.png", triangulation_image)

    if show_steps:
        background = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
        alpha = alpha_image / 255.0
        composite_image = background * (1 - alpha) + image[:, :, 0:3] * alpha
        for tri in triangulation.elements:
            a, b, c = tri
            pa = triangulation.points[a]
            pb = triangulation.points[b]
            pc = triangulation.points[c]
            color = (64, 64, 64)
            cv2.line(composite_image, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), color)
            cv2.line(composite_image, (int(pb[1]), int(pb[0])), (int(pc[1]), int(pc[0])), color)
            cv2.line(composite_image, (int(pc[1]), int(pc[0])), (int(pa[1]), int(pa[0])), color)
        cv2.imwrite(output_prefix + "_composite.png", composite_image)

    vertices = []
    tex_coords = []
    for p in triangulation.points:
        x = p[1] * 1.0
        y = (h - p[0]) * 1.0
        tx = x / w
        ty = y / h
        vertices.append([x, y])
        tex_coords.append([tx, ty])
    triangles = []
    for t in triangulation.elements:
        triangles.append([t[0], t[2], t[1]])

    os.makedirs(output_dir, exist_ok=True)
    output_name, output_extension = os.path.splitext(output_file_name)
    if output_extension == ".msgpck":
        with open(output_file_name, 'wb') as fout:
            output_data = binary_serialize_gladiolus_mesh(
                vertices, tex_coords, triangles, png_file_name, output_file_name)
            msgpack.pack(output_data, fout)
    elif output_extension == ".json":
        with open(output_file_name, 'wt', encoding="utf-8") as fout:
            output_data = readable_serialize_gladiolus_mesh(
                vertices, tex_coords, triangles, png_file_name, output_file_name)
            json_content = json.dumps(output_data, indent=2, ensure_ascii=False)
            fout.write(json_content)


def generate_simple_mesh_file(
        png_file_name: str,
        output_file_name: str,
        dilation: int):
    """
    Generate a simple rectangular mesh that covers a given image.

    :param png_file_name: the input PNG file
    :param output_file_name: name of the output file
    :param dilation: the number of pixels between the mesh boundary and the image
    """
    image = cv2.imdecode(numpy.fromfile(png_file_name, dtype=numpy.uint8), cv2.IMREAD_UNCHANGED)
    assert image.shape[2] == 4
    h, w = image.shape[0], image.shape[1]

    if output_file_name is None:
        input_dir = os.path.dirname(png_file_name)
        base_file_name = os.path.splitext(os.path.basename(png_file_name))[0]
        if input_dir == "":
            output_file_name = base_file_name + "_mesh.msgpck"
        else:
            output_file_name = input_dir + "/" + base_file_name + "_mesh.msgpck"
    output_name, output_extension = os.path.splitext(output_file_name)
    assert output_extension.lower() in ['.msgpck', '.json']

    output_dir = os.path.dirname(output_file_name)

    dilation = int(dilation)
    raw_vertices = [
        [-dilation, -dilation],
        [w + dilation, -dilation],
        [w + dilation, h + dilation],
        [-dilation, h + dilation]
    ]

    vertices = []
    tex_coords = []
    for p in raw_vertices:
        x = p[0] * 1.0
        y = (h - p[1]) * 1.0
        tx = x / w
        ty = y / h
        vertices.append([x, y])
        tex_coords.append([tx, ty])
    triangles = [
        [0, 1, 2],
        [0, 2, 3]
    ]

    os.makedirs(output_dir, exist_ok=True)
    output_name, output_extension = os.path.splitext(output_file_name)
    if output_extension == ".msgpck":
        with open(output_file_name, 'wb') as fout:
            output_data = binary_serialize_gladiolus_mesh(
                vertices, tex_coords, triangles, png_file_name, output_file_name)
            msgpack.pack(output_data, fout)
    elif output_extension == ".json":
        with open(output_file_name, 'wt', encoding="utf-8") as fout:
            output_data = readable_serialize_gladiolus_mesh(
                vertices, tex_coords, triangles, png_file_name, output_file_name)
            json_content = json.dumps(output_data, indent=2, ensure_ascii=False)
            fout.write(json_content)