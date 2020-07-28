import csv


def load_three_step_data_tsv(data_tsv_file_name: str,
                             pose_size: int = 6):
    print("Loading", data_tsv_file_name, "...")
    examples = []
    data_tsv_file = open(data_tsv_file_name)
    tsvreader = csv.reader(data_tsv_file, delimiter='\t')

    for line in tsvreader:
        rest_image_file_name = line[0]
        morph_only_pose = [float(x) for x in line[1:pose_size + 1]]
        morphed_image_file_name = line[pose_size + 1]
        full_pose = [float(x) for x in line[pose_size + 2: 2 * pose_size + 2]]
        posed_image_file_name = line[2 * pose_size + 2]
        visibility_image_file_name = line[2 * pose_size + 3]

        example = [
            rest_image_file_name,
            morph_only_pose,
            morphed_image_file_name,
            full_pose,
            posed_image_file_name,
            visibility_image_file_name
        ]

        examples.append(example)

    data_tsv_file.close()
    print("Loading", data_tsv_file_name, "done!!!")
    return examples
