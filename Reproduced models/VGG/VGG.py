
# coding: utf-8

# In[1]:


# % pylab inline
from pylab import *


# In[2]:


# imports
import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[3]:


def read_images(dir_path):
    
    X = []
    y = []
    
    # define a label map
    labelmap = {
                'airplane': 0,
                'bird': 1,
                'dog': 2,
                'frog': 3,
                'horse': 4,
                'apple': 5,
                'grape': 6,
                'kiwi': 7,
                'lemon': 8,
                'strawberry': 9
               }
    
    directory_list = os.listdir(dir_path)
    # remove OS X's .DS_Store file
    if '.DS_Store' in directory_list:
        directory_list.remove('.DS_Store')
    
    for i, class_name in enumerate(directory_list):
        for j, image_name in enumerate(os.listdir(dir_path+class_name)):
            image_path = dir_path+class_name+'/'+image_name
            image = cv2.imread(image_path)
            X.append(image)
            y.append(labelmap[class_name])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# In[4]:


train_dir_path1 = 'hw2 data/data1/train/'
train_dir_path2 = 'hw2 data/data2/train/'

X1, y1 = read_images(train_dir_path1)
X2, y2 = read_images(train_dir_path2)


# In[5]:


def load_data(dir_path1, dir_path2):
    
    X1, y1 = read_images(dir_path1)
    X2, y2 = read_images(dir_path2)
    
    X2_resized = np.zeros((X2.shape[0], 32, 32, X2.shape[3]), dtype=np.uint8)
    
    for i in range(X2.shape[0]):
        X2_resized[i,:,:,0] = cv2.resize(X2[i,:,:,0], (32,32))
        X2_resized[i,:,:,1] = cv2.resize(X2[i,:,:,1], (32,32))
        X2_resized[i,:,:,2] = cv2.resize(X2[i,:,:,2], (32,32))
    
    X = np.append(X1, X2_resized, axis=0)
    y = np.append(y1, y2, axis=0)
    
    return X, y


# In[6]:


train_dir_path1 = 'hw2 data/data1/train/'
train_dir_path2 = 'hw2 data/data2/train/'
X_train, y_train= load_data(train_dir_path1, train_dir_path2)


# In[7]:


class CIFAR10(torch.utils.data.dataset.Dataset):
    __Xs = None
    __ys = None
    
    def __init__(self, dir_path1, dir_path2, transform=None):
        self.transform = transform
        self.__Xs, self.__ys = load_data(dir_path1, dir_path2)
        
    def __getitem__(self, index):
        img = self.__Xs[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
            
        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]))
        
        return img, label
    
    def __len__(self):
        return self.__Xs.shape[0]


# In[8]:


train_dir_path1 = 'hw2 data/data1/train/'
train_dir_path2 = 'hw2 data/data2/train/'
batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR10(train_dir_path1, train_dir_path2, transform=transform)

num_samples = len(trainset)
indices = list(range(num_samples))
validation_size = int(0.1 * num_samples)
print("Validation set size: " + str(validation_size))

validation_idx = np.random.choice(indices, size=validation_size, replace=False)
train_idx = list(set(indices) - set(validation_idx))

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(validation_idx)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=validation_sampler, num_workers=1)


# In[9]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[10]:


print(device)
torch.cuda.empty_cache()


# In[11]:


cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],}

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        
        self.features = self._make_layers(cfg['VGG16'])
        
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    


net = VGG()
if torch.cuda.is_available():
    print("Running on GPU")
    net = net.cuda()


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


# In[ ]:


# epochs = 250

# training_losses = []
# validation_losses = []
# correct = 0
# total = 0

# for epoch in range(epochs):  # loop over the dataset multiple times

#     training_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         if torch.cuda.is_available():
#             inputs = inputs.cuda()
#             labels = labels.cuda()

#         optimizer.zero_grad()

#         outputs = net(inputs)
#         if torch.cuda.is_available():
#             outputs = outputs.cuda()
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         training_loss += loss.item()
#     training_losses.append(training_loss)
        
#     validation_loss = 0.0
#     for i, data in enumerate(validation_loader, 0):
#         inputs, labels = data
#         if torch.cuda.is_available():
#             inputs = inputs.cuda()
#             labels = labels.cuda()
        
#         outputs = net(inputs)
#         if torch.cuda.is_available():
#             outputs = outputs.cuda()
#         loss = criterion(outputs, labels)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
#         validation_loss += loss.item()
#     validation_losses.append(validation_loss)
    
#     print('epoch %d/%d \t training loss: %.3f \t validation_loss: %.3f \t accuracy: %d%%' %
#               (epoch + 1, epochs, training_loss, validation_loss, 100 * correct / total))

# print('Finished Training')

# torch.save(net.state_dict(), 'VGGNet.pt')
# print("Saved model in VGGNet.pt")


# In[ ]:


test_dir_path1 = 'hw2 data/data1/test/'
test_dir_path2 = 'hw2 data/data2/test/'
batch_size = 1

testset = CIFAR10(test_dir_path1, test_dir_path2, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)


# In[ ]:


correct = 0
total = 0


net = VGG()
net.load_state_dict(torch.load('VGGNet.pt'))

if torch.cuda.is_available():
    net.cuda()


with torch.no_grad():
    for data in test_loader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (total,
    100 * correct / total))


# In[ ]:

# print(training_losses)
# print(validation_losses)

# plt.plot(list(range(epochs)), training_losses, label='training')
# plt.plot(list(range(epochs)), validation_losses, label='validation')
# plt.legend()
# plt.show()

