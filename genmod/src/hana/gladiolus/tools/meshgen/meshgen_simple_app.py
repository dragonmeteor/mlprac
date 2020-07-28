from argparse import ArgumentParser

from hana.gladiolus.tools.meshgen.generate_mesh import generate_simple_mesh_file


def create_argument_parser():
    parser = ArgumentParser(description="Generate a simple rectangular mesh that covers a given image.")
    parser.add_argument("png_file_name", metavar='png_file_name', type=str, help="the input PNG file")
    parser.add_argument("-o", "--output", action="store",
                        help="name of the output file")
    parser.add_argument("-m", "--margin", action="store", default="0",
                        help="the number of pixels betweeen the mesh boundary and the image")
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    generate_simple_mesh_file(args.png_file_name, args.output, args.margin)
