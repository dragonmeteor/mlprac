from argparse import ArgumentParser

from hana.gladiolus.tools.meshgen.generate_mesh import generate_delaunay_mesh_file


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
    generate_delaunay_mesh_file(
        args.png_file_name,
        args.output,
        float(args.area),
        int(args.dilation),
        args.show_steps)
