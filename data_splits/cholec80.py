phase_dict_key = {
    0: 'Preparation', 
    1: 'CalotTriangleDissection', 
    2: 'ClippingCutting', 
    3: 'GallbladderDissection', 
    4: 'GallbladderPackaging', 
    5: 'CleaningCoagulation', 
    6: 'GallbladderRetraction', 
}

"""
Data split of the dataset:
"train": training set
"val": validation set
"test": testing set
"train_all": testing set + validation set, used for distillation
"train_all": all data

"cross_i": mimicking set for Iteration i
"cross_i_train": training set for Iteration i
"""
DATA_SPLIT = {
    "train": [f"video{i:02d}" for i in range(1, 33)],
    "val": [f"video{i:02d}" for i in range(33, 41)],
    "test": [f"video{i:02d}" for i in range(41, 81)],

    "cross_1": [f"video{i:02d}" for i in range(1, 9)],
    "cross_2": [f"video{i:02d}" for i in range(9, 17)],
    "cross_3": [f"video{i:02d}" for i in range(17, 25)],
    "cross_4": [f"video{i:02d}" for i in range(25, 33)],
    "cross_5": [f"video{i:02d}" for i in range(33, 41)],
}
DATA_SPLIT["train_all"] = DATA_SPLIT["train"] + DATA_SPLIT["val"]
DATA_SPLIT["all"] = DATA_SPLIT["train"] + DATA_SPLIT["val"] + DATA_SPLIT["test"]
DATA_SPLIT["cross_1_train"] = DATA_SPLIT["cross_2"] + DATA_SPLIT["cross_3"] + DATA_SPLIT["cross_4"] + DATA_SPLIT["cross_5"]
DATA_SPLIT["cross_2_train"] = DATA_SPLIT["cross_1"] + DATA_SPLIT["cross_3"] + DATA_SPLIT["cross_4"] + DATA_SPLIT["cross_5"]
DATA_SPLIT["cross_3_train"] = DATA_SPLIT["cross_1"] + DATA_SPLIT["cross_2"] + DATA_SPLIT["cross_4"] + DATA_SPLIT["cross_5"]
DATA_SPLIT["cross_4_train"] = DATA_SPLIT["cross_1"] + DATA_SPLIT["cross_2"] + DATA_SPLIT["cross_3"] + DATA_SPLIT["cross_5"]
DATA_SPLIT["cross_5_train"] = DATA_SPLIT["cross_1"] + DATA_SPLIT["cross_2"] + DATA_SPLIT["cross_3"] + DATA_SPLIT["cross_4"]


"""
get_label: used for the Dataset
input: path to the annotation file
output: a python dict with key: frame index, value: corresponding label
"""
def get_label(label_path):
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