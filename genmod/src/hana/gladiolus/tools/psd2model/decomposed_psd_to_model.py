import json
import os

from hana.gladiolus.tools.meshgen.generate_mesh import generate_simple_mesh_file, generate_delaunay_mesh_file


def decomposed_psd_to_model(
        json_file_name: str,
        output_dir: str,
        dilation: int = 0,
        maximum_area: float = 10,
        mode: str = 'delaunay'):
    with open(json_file_name, "rt") as fin:
        layers_data = json.load(fin)
    for layer in layers_data['layers']:
        if layer['type'] != 'layer':
            continue
        dir, file_name = os.path.split(layer['file_name'])
        base_name, ext = os.path.splitext(file_name)
        output_file_name = output_dir + "/" + base_name + "_mesh.msgpck"
        if mode == 'simple':
            generate_simple_mesh_file(png_file_name=layer['file_name'],
                                      output_file_name=output_file_name,
                                      dilation=dilation)
        else:
            generate_delaunay_mesh_file(png_file_name=layer['file_name'],
                                        output_file_name=output_file_name,
                                        dilation=dilation,
                                        maximum_area=maximum_area)


if __name__ == "__main__":
    decomposed_psd_to_model(
        json_file_name="data/gladiolus/_20200219/live2d-kyoukasyo-slime/layers.json",
        output_dir="data/gladiolus/_20200219/live2d-kyoukasyo-slime",
        dilation=0,
        maximum_area=10.0,
        mode='simple')
