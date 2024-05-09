import torch

# acclelerate data loading
class Prefetcher(): 
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