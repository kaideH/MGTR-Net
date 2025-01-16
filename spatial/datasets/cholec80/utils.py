import random, numbers
import torch


phase_dict_key = {
    'Preparation': 0, 
    'CalotTriangleDissection': 1, 
    'ClippingCutting': 2, 
    'GallbladderDissection': 3, 
    'GallbladderPackaging': 4, 
    'CleaningCoagulation': 5, 
    'GallbladderRetraction': 6
}

class AccimageListRandomResizeCrop(object):

    def __init__(self, size=224, scale=(0.8, 0.95), ratio=(1.5, 1.8)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, imgs): # imgs, list of pil image, w < h
        scale = random.uniform(*self.scale)
        ratio = random.uniform(*self.ratio)

        h, w = imgs[0].size
        tw = int(w * scale) # w is shorter
        th = int(tw * ratio)

        h1 = random.randint(0, h - th)
        w1 = random.randint(0, w - tw)
        crop_imgs = [ img.crop(box=(h1, w1, h1 + th, w1 + tw)) for img in imgs ]
        new_imgs = [img.resize(self.size) for img in crop_imgs]
        return new_imgs
    

def get_label(label_path):
        label_dict = {}
        f = open(label_path)
        lines = f.readlines()[1:] # first line is "Frame\tPhase"
        f.close()
        for line in lines:
            frame_idx, label = line.strip().split("\t")
            label_dict[int(frame_idx)] = phase_dict_key[label]
        return label_dict


class Prefetcher(): # acclelerate data loading
    def __init__(self, loader):
        self.data_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        self.inputs = None
        self.labels = None
        self.video_name = None
        self.frame_idx = None

        self.preload()

    def preload(self):
        try:
            self.inputs, self.labels, self.video_name, self.frame_idx = next(self.loader)
        except StopIteration:
            self.inputs = None
            self.labels = None
            self.video_name = None
            self.frame_idx = None
            self.data_loader.dataset.reset_buffer()
            self.loader = iter(self.data_loader)
            return

        with torch.cuda.stream(self.stream):
            self.inputs = self.inputs.cuda(non_blocking=True).float()
            self.labels = self.labels.cuda(non_blocking=True).long()

    def get_next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.inputs
        labels = self.labels
        video_name = self.video_name
        frame_idx = self.frame_idx

        if inputs is not None:
            inputs.record_stream(torch.cuda.current_stream())
        if labels is not None:
            labels.record_stream(torch.cuda.current_stream())
            
        self.preload()
        return inputs, labels, video_name, frame_idx

    def __len__(self):
        return len(self.loader)
    
    
if __name__ == "__main__":
    from pprint import pprint
    pprint(data_sub_set)




