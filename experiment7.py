import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable

#准备数据集
train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#VGG16
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            #conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

#创建vgg16模型
model = VGG16()
if torch.cuda.is_available():
    model = model.cuda()

#模型训练
max_epoches = 10
epoch = 0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

while epoch < max_epoches:

    model.train()
    correct = 0
    for batch_idx, (data, label) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
            data = Variable(data)
            label = Variable(label)
            out = model(data)
            loss = criterion(out, label)
            _, pred = torch.max(out, dim=1)
            correct += (pred == label).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch += 1
    print(f'Epoch : {epoch}, Train, Accuracy : {correct / len(train_dataset)}')

    #每轮进行一次模型评估
    model.eval()
    eval_correct = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_dataloader):
            data = data.cuda()
            label = label.cuda()
            data = Variable(data)
            label = Variable(label)
            out = model(data)
            _, pred = torch.max(out, dim=1)
            eval_correct += (pred == label).sum().item()
    print(f'Epoch : {epoch}, Eval, Accuracy : {eval_correct / len(test_dataset)}')



