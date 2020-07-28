import os
from argparse import ArgumentParser

import cv2
import numpy
from scipy.spatial.qhull import ConvexHull

import meshpy.triangle


def create_argument_parser():
    parser = ArgumentParser(description="Generate a mesh that covers a given image.")
    parser.add_argument("png_file_name", metavar='png_file_name', type=str, help="the input PNG file")
    parser.add_argument("-o", "--output", action="store",
                        help="name of the output file")
    parser.add_argument("-a", "--area", action="store", default="180",
                        help="the maximum area of each triangle")
    parser.add_argument("-d", "--dilation", action="store", default="20",
                        help="size of the kernel of the dilation operator")
    parser.add_argument("--show_steps", action="store_const", const=True, default=False,
                        help="whether to generate images of intermediate steps")
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()

    png_file_name = args.png_file_name

    image = cv2.imread(png_file_name, cv2.IMREAD_UNCHANGED)
    assert image.shape[2] == 4
    h, w = image.shape[0], image.shape[1]

    if args.output is None:
        input_dir = os.path.dirname(png_file_name)
        base_file_name = os.path.splitext(os.path.basename(png_file_name))[0]
        if input_dir == "":
            args.output = base_file_name + "_mesh.txt"
        else:
            args.output = input_dir + "/" + base_file_name + "_mesh.txt"

    output_dir = os.path.dirname(args.output)
    output_base_file_name = os.path.splitext(os.path.basename(args.output))[0]
    if output_dir == "":
        output_prefix = output_base_file_name
    else:
        output_prefix = output_dir + "/" + output_base_file_name

    alpha_image = image[:, :, 3].reshape(h, w, 1)
    if args.show_steps:
        cv2.imwrite(output_prefix + "_alpha.png", alpha_image)

    dilation_kernel_size = int(args.dilation)
    kernel = numpy.ones((dilation_kernel_size, dilation_kernel_size), dtype=numpy.uint8)
    dilated_image = cv2.dilate(alpha_image, kernel, iterations=1)
    if args.show_steps:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_prefix + "_dilated.png", dilated_image)

    dilated_non_zero_image = numpy.not_equal(dilated_image, 0).astype(numpy.uint8) * 255
    if args.show_steps:
        cv2.imwrite(output_prefix + "_dilated_non_zero.png", dilated_non_zero_image)

    nonzero_indices = numpy.nonzero(dilated_non_zero_image.reshape(h, w))
    nonzero_indices = numpy.stack(nonzero_indices, axis=1)
    convex_hull = ConvexHull(nonzero_indices)
    convex_hull_vertex_indices = convex_hull.vertices
    convex_hull_vertices = nonzero_indices[convex_hull_vertex_indices] + 0.5
    n = convex_hull_vertices.shape[0]
    if args.show_steps:
        convex_hull_image = numpy.zeros((h, w, 3), numpy.uint8)
        for i0 in range(n):
            i1 = (i0 + 1) % n
            p0 = convex_hull_vertices[i0]
            p1 = convex_hull_vertices[i1]
            cv2.line(convex_hull_image, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), (255, 255, 255))
        cv2.imwrite(output_prefix + "_convex_hull.png", convex_hull_image)

    max_area = float(args.area)
    convex_hull_mesh_info = meshpy.triangle.MeshInfo()
    convex_hull_mesh_info.set_points(convex_hull_vertices.tolist())
    facets = [[i, (i + 1) % n] for i in range(n)]
    convex_hull_mesh_info.set_facets(facets)
    triangulation = meshpy.triangle.build(convex_hull_mesh_info, max_volume=max_area, volume_constraints=True)
    if args.show_steps:
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

    if args.show_steps:
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

    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "wt") as fout:
        fout.write("%d %d\n" % (len(triangulation.points), len(triangulation.elements)))
        for p in triangulation.points:
            fout.write("%f %f\n" % (p[1], p[0]))
        for t in triangulation.elements:
            fout.write("%d %d %d\n" % (t[0], t[1], t[2]))
