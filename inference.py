import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from duochidu_vig import pvig_s_224_gelu
from gcn_lib import act_layer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class GraphDataset_test(Dataset):
    def __init__(self, data_tensor, label_list, file_names=None):
        self.data = data_tensor
        self.label = label_list
        self.file_names = file_names if file_names else [""] * len(data_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.file_names[idx]



def load_dataset(file_path300, file_path700, file_path1200, duochidu=3):
    data_list = []
    label_list = []
    file_name_list = []

    for file_name in os.listdir(file_path300):
        if file_name.endswith(".txt"):
            duochidu_list = []
            for i in range(duochidu):
                if i == 0:
                    path = os.path.join(file_path300, file_name)
                elif i == 1:
                    path = os.path.join(file_path700, file_name)
                elif i == 2:
                    path = os.path.join(file_path1200, file_name)

                points = np.loadtxt(path, dtype=np.float32)
                x = torch.from_numpy(points).float()
                duochidu_list.append(x)

            data_list.append(duochidu_list)
            tag = file_name.split('-')[0]
            label_list.append(int(tag))
            file_name_list.append(file_name)

    dataset = GraphDataset_test(data_list, label_list, file_name_list)
    return dataset


def inference(model, prediction, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    incorrect_images = []

    with torch.no_grad():
        for data, labels, file_names in dataloader:
            data_duochidu_list = []
            for data_duochidu in data:
                data_duochidu = data_duochidu.to(device)
                data_duochidu = torch.transpose(data_duochidu, 1, 2)
                output = model(data_duochidu)
                data_duochidu_list.append(output)

            output_cat = torch.cat(data_duochidu_list, dim=1)
            output_cat = prediction(output_cat)
            output_cat = output_cat.squeeze(-1)

            preds = output_cat.argmax(axis=1)

            labels = torch.tensor(labels).to(device)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    incorrect_images.append((file_names[i], labels[i].item(), preds[i].item()))

    return all_labels, all_preds, incorrect_images



if __name__ == "__main__":
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


    model, _ = pvig_s_224_gelu()
    prediction = nn.Sequential(
        nn.Conv1d(640*3, 1024, 1, bias=True),
        nn.BatchNorm1d(1024),
        act_layer('gelu'),
        nn.Dropout(0.0),
        nn.Conv1d(1024, 70, 1, bias=True)
    )

    model.to(device)
    prediction.to(device)

    # 
    
    #checkpoint
    checkpoint = torch.load("/data1/jinming/jinming/BASR-GCN-MS/GBASR_mpeg7_model.pt", map_location=device)
    print(1)
    print(checkpoint.keys())
    print(1)

    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    prediction.load_state_dict(checkpoint['prediction'], strict=True)

    file_path300_test = "/data1/jinming/new_datasets/MPEG7_10.23_norm_test_global_reduce300_gujia_jiaodu/"
    file_path700_test = "/data1/jinming/new_datasets/MPEG7_10.23_norm_test_global_reduce700_gujia_jiaodu/"
    file_path1200_test = "/data1/jinming/new_datasets/MPEG7_10.23_norm_test_global_reduce1200_gujia_jiaodu/"

    test_dataset = load_dataset(file_path300_test, file_path700_test, file_path1200_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    y_true, y_pred, incorrect = inference(model, prediction, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print(f"\n Accuracy: {acc:.4f}")
    print("\n Confusion Matrix:")
    print(cm)
    print("\n Classification Report:")
    print(report)

    print("\n Incorrect Predictions (filename, true, pred):")
    for item in incorrect[:20]:
        print(item)
