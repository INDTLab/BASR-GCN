import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from torchvision.transforms.functional import resize
from sklearn.model_selection import train_test_split

from torch.nn import Sequential as Seq

from duochidu_vig import pvig_s_224_gelu
from sklearn.metrics import confusion_matrix, classification_report
from gcn_lib import Grapher, act_layer
from torch.utils.data import Dataset, DataLoader, random_split
import os.path
import cv2
import numpy as np

model,opt = pvig_s_224_gelu()
model.train()




class GraphDataset(Dataset):
    def __init__(self, data_tensor, label_list):
        self.data = data_tensor
        self.label = label_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class GraphDataset_test(Dataset):
    def __init__(self, data_tensor, label_list, file_names=None):
        self.data = data_tensor
        self.label = label_list
        self.file_names = file_names if file_names else [""] * len(data_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.file_names[idx]

device = torch.device("cuda:7")

data_list_train = []
label_list_train = []

duochidu = 3



file_path300='/data1/jinming/new_datasets/MPEG7_10.23_norm_train_global_reduce300_gujia_jiaodu/'
file_path700='/data1/jinming/new_datasets/MPEG7_10.23_norm_train_global_reduce700_gujia_jiaodu//'
file_path1200='/data1/jinming/new_datasets/MPEG7_10.23_norm_train_global_reduce1200_gujia_jiaodu/'



for file_name in os.listdir(file_path300):
    if file_name.endswith(".txt"):

        duochidu_list_train=[]
        for i in  range(duochidu):
            if i == 0:
                img = os.path.join(file_path300, file_name)
                points = np.loadtxt(img, dtype=np.float32)
                x = np.array(points)
                x = torch.from_numpy(x).float()
                x = torch.tensor(x)
                duochidu_list_train.append(x)
            if i == 1:
                img = os.path.join(file_path700, file_name)
                points = np.loadtxt(img, dtype=np.float32)
                x = np.array(points)
                x = torch.from_numpy(x).float()
                x = torch.tensor(x)
                duochidu_list_train.append(x)
            if i == 2:
                img = os.path.join(file_path1200, file_name)
                points = np.loadtxt(img, dtype=np.float32)
                x = np.array(points)
                x = torch.from_numpy(x).float()
                x = torch.tensor(x)
                duochidu_list_train.append(x)

        data_list_train.append(duochidu_list_train)

        # data = load_data(img,num_node)
        # print(data)
        tag = file_name.split('-')[0]

        label_list_train.append(int(tag))




data_list_test = []
label_list_test = []


file_name_list_test = []



file_path300_test='/data1/jinming/new_datasets/MPEG7_10.23_norm_test_global_reduce300_gujia_jiaodu/'
file_path700_test='/data1/jinming/new_datasets/MPEG7_10.23_norm_test_global_reduce700_gujia_jiaodu/'
file_path1200_test='/data1/jinming/new_datasets/MPEG7_10.23_norm_test_global_reduce1200_gujia_jiaodu/'

for file_name in os.listdir(file_path300_test):
    if file_name.endswith(".txt"):

        duochidu_list_test = []
        for i in range(duochidu):
            if i == 0:
                img = os.path.join(file_path300_test, file_name)
                points = np.loadtxt(img, dtype=np.float32)
                x = np.array(points)
                x = torch.from_numpy(x).float()
                x = torch.tensor(x)
                duochidu_list_test.append(x)
            if i == 1:
                img = os.path.join(file_path700_test, file_name)
                points = np.loadtxt(img, dtype=np.float32)
                x = np.array(points)
                x = torch.from_numpy(x).float()
                x = torch.tensor(x)
                duochidu_list_test.append(x)
            if i == 2:
                img = os.path.join(file_path1200_test, file_name)
                points = np.loadtxt(img, dtype=np.float32)
                x = np.array(points)
                x = torch.from_numpy(x).float()
                x = torch.tensor(x)
                duochidu_list_test.append(x)

        data_list_test.append(duochidu_list_test)

        # data = load_data(img,num_node)
        # print(data)
        tag = file_name.split('-')[0]

        label_list_test.append(int(tag))
        
        file_name_list_test.append(file_name)
        # labels.append(tag)




dataset_train = GraphDataset(data_list_train, label_list_train)

#dataset_test = GraphDataset(data_list_test, label_list_test)
dataset_test = GraphDataset_test(data_list_test, label_list_test,file_name_list_test)
train_dataset = dataset_train

test_dataset = dataset_test

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# train_dataset, test_dataset =  train_test_split(dataset, test_size=0.5,random_state=42)

# val_dataset, test_dataset1 = random_split(test_dataset, [val_size, test_size1])

val_dataset, test_dataset1 = train_test_split(test_dataset, test_size=0.5, random_state=42)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

train_data_list = train_loader
test_data_list = test_loader
val_data_list = val_loader

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)





model.to(device)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)


prediction = Seq(nn.Conv1d(640*3, 1024, 1, bias=True),
                              nn.BatchNorm1d(1024),
                              act_layer('gelu'),
                              nn.Dropout(0.0),
                              nn.Conv1d(1024, 70, 1, bias=True))
prediction=prediction.to(device)
def val():
    val_acc = 0
    correct = 0

    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    incorrect_images = []
    model.eval()
    with torch.no_grad():
        for data, labels,file_names in val_loader:

            data_duochidu_list=[]
            for data_duochidu in data:
    
                data_duochidu = data_duochidu.to(device)
                data_duochidu=torch.transpose(data_duochidu, 1, 2)
                output = model(data_duochidu)
                data_duochidu_list.append(output)

            output_cat = torch.cat(data_duochidu_list, dim=1)
            #output_cat=output_cat.squeeze(-1)
            #output_cat=torch.transpose(output_cat, 1, 2)
            
            
            output_cat =prediction(output_cat)
            output_cat=output_cat.squeeze(-1)
            
            #output_cat=torch.transpose(output_cat, 1, 2)
            pred = output_cat.argmax(axis=1)

            # correct += pred.eq(labels).sum().item()

            labels = torch.tensor(labels)
            labels = labels.to(device)
            total_samples += labels.size(0)
            total_correct += (pred == labels).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for i in range(len(labels)):
                if pred[i] != labels[i]:
                    incorrect_images.append((file_names[i], labels[i].item(), pred[i].item()))
            # acc(output,data.y,correct)
    # print('correct numbei', correct)
    # val_acc = correct / len(val_loader)

    val_acc = total_correct / total_samples

    return val_acc,all_preds,all_labels,incorrect_images


max_acc = 0.0
max_epoch = 0

for epoch in range(500):
    
    total_loss = 0.0
    total_acc = 0.0

    for data, label in train_loader:
        model.train()
        data_duochidu_list = []
        for data_duochidu in data:
            data_duochidu = data_duochidu.to(device)
            data_duochidu=torch.transpose(data_duochidu, 1, 2)
            output = model(data_duochidu)
            
            
            
            data_duochidu_list.append(output)

        output_cat = torch.cat(data_duochidu_list, dim=1)
        
        #output_cat=output_cat.squeeze(-1)
        #output_cat=output_cat.reshape(-1,1920)
        #print(f"output_cat:{output_cat.shape}")
        output_cat =prediction(output_cat)
        # x, y = data.x, data.y
        output_cat=output_cat.squeeze(-1)
        label = torch.tensor(label)
        label = label.to(device)

        optimizer.zero_grad()



        loss = criterion(output_cat, label)

        loss.backward()
        optimizer.step()

        acc = accuracy(output_cat, label)

        total_loss += loss.item()
        total_acc += acc

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    print(f"Epoch {epoch + 1}, loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")

    acc,all_preds,all_labels,incorrect_images =   val()
    
    val_acc=acc

    print(f'Val Accuracy: {val_acc:.4f},best Accuracy: {max_acc:.4f}')

    if val_acc >= max_acc:
        max_acc = val_acc
        max_epoch = epoch
        
        

print(f"BEST Accuracy: {max_acc}")


