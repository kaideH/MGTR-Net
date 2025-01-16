## Cholec80
data_sub_set = {
    "train": [f"video{i:02d}" for i in range(1, 33)],
    "val": [f"video{i:02d}" for i in range(33, 41)],
    "test": [f"video{i:02d}" for i in range(41, 81)],

    "cross_1": [f"video{i:02d}" for i in range(1, 9)],
    "cross_2": [f"video{i:02d}" for i in range(9, 17)],
    "cross_3": [f"video{i:02d}" for i in range(17, 25)],
    "cross_4": [f"video{i:02d}" for i in range(25, 33)],
    "cross_5": [f"video{i:02d}" for i in range(33, 41)],
}
data_sub_set["train_all"] = data_sub_set["train"] + data_sub_set["val"]
data_sub_set["all"] = data_sub_set["train"] + data_sub_set["val"] + data_sub_set["test"]
data_sub_set["cross_1_train"] = data_sub_set["cross_2"] + data_sub_set["cross_3"] + data_sub_set["cross_4"] + data_sub_set["cross_5"]
data_sub_set["cross_2_train"] = data_sub_set["cross_1"] + data_sub_set["cross_3"] + data_sub_set["cross_4"] + data_sub_set["cross_5"]
data_sub_set["cross_3_train"] = data_sub_set["cross_1"] + data_sub_set["cross_2"] + data_sub_set["cross_4"] + data_sub_set["cross_5"]
data_sub_set["cross_4_train"] = data_sub_set["cross_1"] + data_sub_set["cross_2"] + data_sub_set["cross_3"] + data_sub_set["cross_5"]
data_sub_set["cross_5_train"] = data_sub_set["cross_1"] + data_sub_set["cross_2"] + data_sub_set["cross_3"] + data_sub_set["cross_4"]
data_sub_set_cholec80 = data_sub_set


def get_label_cholec80(label_path):
    phase_dict_key = {
        'Preparation': 0, 
        'CalotTriangleDissection': 1, 
        'ClippingCutting': 2, 
        'GallbladderDissection': 3, 
        'GallbladderPackaging': 4, 
        'CleaningCoagulation': 5, 
        'GallbladderRetraction': 6
    }

    label_dict = {}
    f = open(label_path)
    lines = f.readlines()[1:] # first line is "Frame\tPhase"
    f.close()
    for line in lines:
        frame_idx, label = line.strip().split("\t")
        label_dict[int(frame_idx)] = phase_dict_key[label]
    return label_dict