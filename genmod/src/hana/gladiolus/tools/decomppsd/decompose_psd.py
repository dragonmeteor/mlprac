import json
import os

from PIL import ImageOps
from psd_tools import PSDImage


class DecomposePsdParams:
    def __init__(self,
                 pdf_file_name: str,
                 output_dir: str,
                 extra_margin: int = 1):
        self.extra_margin = extra_margin
        self.output_dir = output_dir
        self.psd_file_name = pdf_file_name
        assert self.extra_margin >= 1

def expand_bbox(bbox, extra_margin):
    return [
        bbox[0],
        bbox[1],
        bbox[2] + 2 * extra_margin,
        bbox[3] + 2 * extra_margin
    ]


def process_layer(layer, psd_info: dict, parent_index: int, params: DecomposePsdParams):
    layer_info = {}
    layer_info["name"] = layer.name
    layer_info["parent_index"] = parent_index
    psd_info["layers"].append(layer_info)
    current_index = len(psd_info["layers"]) - 1
    layer_info["bbox"] = expand_bbox(layer.bbox, params.extra_margin)
    layer_info["visible"] = layer.visible

    if layer.is_group():
        layer_info["type"] = "group"
        for child in layer:
            process_layer(child, psd_info, current_index, params)
    else:
        layer_info["type"] = "layer"
        composed_layer = ImageOps.expand(layer.compose(), border=params.extra_margin, fill=0)
        if composed_layer is not None:
            file_name = params.output_dir + ("/%08d" % current_index) + "-" + layer.name + ".png"
            layer_info["file_name"] = file_name
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            composed_layer.save(file_name)


def decompose_psd(params: DecomposePsdParams):
    psd = PSDImage.open(params.psd_file_name)

    psd_info = {}
    psd_info['width'] = psd.width + params.extra_margin * 2
    psd_info['height'] = psd.height + params.extra_margin * 2
    psd_info['layers'] = []

    for layer in psd:
        process_layer(layer, psd_info, -1, params)

    json_content = json.dumps(psd_info, indent=2, ensure_ascii=False)
    json_file_name = params.output_dir + "/layers.json"
    os.makedirs(os.path.dirname(json_file_name), exist_ok=True)
    with open(json_file_name, "wt", encoding="utf-8") as fout:
        fout.write(json_content)
