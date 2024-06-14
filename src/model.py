import torch
from torch import nn

class MHV(nn.Module):
    def __init__(self):
        super(MHV, self).__init__()
        self.embedding = nn.Embedding(256, 512)
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(1024, 1193, 3, 1, 1),
            nn.BatchNorm2d(1193),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.convTrans = nn.ConvTranspose2d(1193, 1193, 8, stride=4, padding=2)
    
    def forward(self, x):
        x = self.embedding(x) # torch.Size([1, 3, 224, 224, 1024])
        x = x.permute(0, 4, 1, 2, 3)  # 調整維度順序以配合卷積層輸入
        x1, x2, x3 = x[:, :, 0, :, :], x[:, :, 1, :, :], x[:, :, 2, :, :]
        # (1, 512, 224, 224)
        x1 = self.convBlock1(x1)
        x2 = self.convBlock1(x2)
        x3 = self.convBlock1(x3)
        # (1, 1024, 112, 112)
        x1 = self.convBlock2(x1)
        x2 = self.convBlock2(x2)
        x3 = self.convBlock2(x3)
        # (1, 1193, 56, 56)
        x1 = self.convTrans(x1)
        x2 = self.convTrans(x2)
        x3 = self.convTrans(x3)
        # (1, 1193, 224, 224)
        x = torch.cat((x1, x2, x3), 1)
        return x

if __name__ == '__main__': # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MHV()
    test_tensor = torch.randint(0, 256, (1, 3, 224, 224))
    model.to(device)
    test_tensor = test_tensor.to(device)
    print(test_tensor.shape)
    output = model(test_tensor)
    print(output.shape)