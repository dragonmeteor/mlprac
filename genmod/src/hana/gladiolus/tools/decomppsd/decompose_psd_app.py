from argparse import ArgumentParser

from hana.gladiolus.tools.decomppsd.decompose_psd import DecomposePsdParams, decompose_psd


def create_argument_parser():
    parser = ArgumentParser(description="Decompose a PSD files into layers.")
    parser.add_argument("psd_file_name", metavar='psd_file_name', type=str, help="the input PSD file")
    parser.add_argument("-o", "--output_dir", action="store",
                        help="the output directory")
    parser.add_argument("-m", "--extra_margin", action="store", default="1",
                        help="the extra margin around each layer image")
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    params = DecomposePsdParams(
        args.psd_file_name,
        args.output_dir,
        int(args.extra_margin))
    decompose_psd(params)
