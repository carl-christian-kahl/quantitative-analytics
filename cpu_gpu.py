import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ABC(torch.nn.Module):
    def __init__(self):
        super(ABC, self).__init__()
        self.a = torch.tensor([0.2])

a = ABC()

a.to(device=device)