import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.c = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        
        self.f1 = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.f2 = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        z = self.c(x)
        z = z.mean(dim=[-1,-2])
        h1 = self.f1(z)
        h2 = self.f2(z)
        return torch.cat([h1, h2], 1)


class Predictor:
    def __init__(self):
        self.WC = np.load("models/cobalt_weights.npy")
        self.net = Net()
        self._load_net()
        self.net.eval()
    
    def _load_net(self, path="models/cnn.pth"):
        self.net.load_state_dict(torch.load(path))

    def cubic_exp(self, iron, thorium):
        return np.array([
            1, 
            iron, 
            thorium, 
            iron*thorium, 
            iron**2, 
            thorium**2,
            iron*thorium**2,
            iron**2*thorium,
            iron**3,
            thorium**3
            ])

    def predict(self, image_arr):
        image_tensor = torch.from_numpy(image_arr).unsqueeze(0).float().permute(0, 3, 1, 2)/255
        with torch.no_grad():
            pred = self.net(image_tensor)
        pred_np = pred.numpy()[0]
        iron, thor = pred_np[0], pred_np[1]
        temp = self.cubic_exp(iron, thor)
        cobt = temp @ self.WC
        return iron, thor, cobt
        