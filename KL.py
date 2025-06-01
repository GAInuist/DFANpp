import pickle
import scipy.io as sio
from sklearn.model_selection import train_test_split
from network.ETN import ETN
from dfan.DFAN import DFAN
from scipy.special import kl_div
import torch.nn.functional as F
from utils import *
from config import *
from data import *
from collections import defaultdict
import random
from TransZero.TransZero import TransZero
import argparse
import json

args = get_config_parser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
# random seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
    print('seed: ' + str(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
pref = str(args.seed)
args.pref = pref
torch.backends.cudnn.benchmark = True

ROOT = args.DATASET_path
DATA_DIR = f'{args.xlsa17_path}/data/{args.DATASET}'
data = sio.loadmat(f'{DATA_DIR}/res101.mat')
# data consists of files names
attrs_mat = sio.loadmat(f'{DATA_DIR}/att_splits.mat')
# attrs_mat is the attributes (class-level information)
image_files = data['image_files']

if args.DATASET == 'AWA2':
    image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])
elif args.DATASET == 'APY':
    image_files = np.array([im_f[0][0].split('APY/')[-1] for im_f in image_files])
else:
    image_files = np.array([im_f[0][0].split('images/')[-1] for im_f in image_files])

# labels are indexed from 1 as it was done in Matlab, so 1 subtracted for Python
labels = data['labels'].squeeze().astype(np.int64) - 1
train_idx = attrs_mat['train_loc'].squeeze() - 1
val_idx = attrs_mat['val_loc'].squeeze() - 1
trainval_idx = attrs_mat['trainval_loc'].squeeze() - 1
test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1

# consider the train_labels and val_labels
train_labels, val_labels = labels[train_idx], labels[val_idx]

# split train_idx to train_idx (used for training) and val_seen_idx
train_idx, val_seen_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels)
# split val_idx to val_idx (not useful) and val_unseen_idx
val_unseen_idx = train_test_split(val_idx, test_size=0.2, stratify=val_labels)[1]
class_name = attrs_mat['allclasses_names']
for idx, _ in enumerate(class_name):
    class_name[idx] = class_name[idx][0][0][4:]
# attribute matrix
attrs_mat = attrs_mat["att"].astype(np.float32).T

### used for validation
# train files and labels
train_files, train_labels = image_files[train_idx], labels[train_idx]
uniq_train_labels, train_labels_based0, counts_train_labels = np.unique(train_labels, return_inverse=True,
                                                                        return_counts=True)
# val seen files and labels
val_seen_files, val_seen_labels = image_files[val_seen_idx], labels[val_seen_idx]
uniq_val_seen_labels = np.unique(val_seen_labels)
# val unseen files and labels
val_unseen_files, val_unseen_labels = image_files[val_unseen_idx], labels[val_unseen_idx]
uniq_val_unseen_labels = np.unique(val_unseen_labels)

### used for testing
# trainval files and labels
trainval_files, trainval_labels = image_files[trainval_idx], labels[trainval_idx]
uniq_trainval_labels, trainval_labels_based0, counts_trainval_labels = np.unique(trainval_labels, return_inverse=True,
                                                                                 return_counts=True)
# test seen files and labels
test_seen_files, test_seen_labels = image_files[test_seen_idx], labels[test_seen_idx]
uniq_test_seen_labels = np.unique(test_seen_labels)
# test unseen files and labels
test_unseen_files, test_unseen_labels = image_files[test_unseen_idx],labels[test_unseen_idx]
uniq_test_unseen_labels = np.unique(test_unseen_labels)

if args.use_w2v:
    w2v_path = f'./w2v/{args.DATASET}_attribute.pkl'
    with open(w2v_path, 'rb') as f:
        w2v = np.array(pickle.load(f))
        w2v = torch.from_numpy(w2v).float().cuda()

testTransform = get_transform(args)
def sample_and_process(data_files, data_labels, num_classes=5):
    grouped_data = defaultdict(list)
    for file, label in zip(data_files, data_labels):
        grouped_data[label].append(file)
    grouped_data = {label: files for label, files in grouped_data.items() if len(files) >= 40}
    sampled_classes = random.sample(list(grouped_data.keys()), num_classes)
    images, labels = [], []
    for cls in sampled_classes:
        for image_file in grouped_data[cls]:
            if "JPEGImages" in image_file:
                image_file = image_file.split('VOCdevkit/')[-1]
            image = Image.open(os.path.join(ROOT, image_file)).convert("RGB")
            image = testTransform(image)
            images.append(image)
            labels.append(cls)
    images_tensor = torch.stack(images)  # [N, C, H, W]
    return images_tensor, labels

# seen_images, seen_labels = sample_and_process(train_files, train_labels, num_classes=4)
unseen_images, unseen_labels = sample_and_process(test_unseen_files, test_unseen_labels, num_classes=50)
print('preprocess data completely')
# print(f"Seen Data: Images Shape: {seen_images.shape}, Labels Shape: {seen_labels.shape}")
# print(f"Unseen Data: Images Shape: {unseen_images.shape}, Labels Shape: {unseen_labels.shape}")

model = ETN(dim=args.v_embedding, attr_num=args.attr_num, drop_rate=args.drop_rate, n_head=args.n_head, word_vector_length=args.word_embedding_length).cuda()
if args.pretrain_path is not None:
    pth_dict = torch.load(args.pretrain_path, map_location='cuda:0')
    model.load_state_dict(pth_dict)

dfan = DFAN(attr_num=args.attr_num).cuda()
dfan.load_state_dict(torch.load(rf"D:\Pretrain_Model\DFAN_Checkpoint\{args.DATASET}_GZSL_DFAN_BEST.pth"))

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=f'TransZero_config/{args.DATASET}_GZSL.json')
config = parser.parse_args()
with open(config.config, 'r') as f:
    config.__dict__ = json.load(f)
transzero = TransZero(config)
# load parameters
model_dict = transzero.state_dict()
saved_dict = torch.load(config.saved_model)
saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
model_dict.update(saved_dict)
transzero.load_state_dict(model_dict)
transzero.to(config.device)

# 自定义 batch 处理函数
def custom_batch_loader(images, labels, batch_size):
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        yield batch_images, batch_labels


def process_data(images, labels, model, dfan, transzero, batch_size):
    model_outputs, dfan_outputs, transzero_outputs = [], [], []
    for batch_images, _ in custom_batch_loader(images, labels, batch_size):
        batch_images = batch_images.cuda()
        dfan_local_prediction = dfan(batch_images)
        dfan_outputs.append(dfan_local_prediction.detach().cpu())

        ETN_local_pred = model(batch_images, w2v)
        model_outputs.append(ETN_local_pred.detach().cpu())

        transzero_pred_package = transzero(batch_images)
        transzero_pred = transzero_pred_package['dazle_embed']
        transzero_outputs.append(transzero_pred.detach().cpu())

        torch.cuda.empty_cache()
    return torch.cat(model_outputs, 0), torch.cat(dfan_outputs, 0), torch.cat(transzero_outputs, 0)

batch_size = 8
etn_outputs, dfan_outputs, transzero_outputs = process_data(unseen_images, unseen_labels, model, dfan, transzero, batch_size)
unique_labels = np.unique(unseen_labels)
kl_e, kl_d, kl_t = [], [], []
for uni_label in unique_labels:
    etn_p, dfan_p, transzero_p = [], [], []
    for etn_sample, dfan_sample, transzero_sample, label in zip(etn_outputs, dfan_outputs, transzero_outputs, unseen_labels):
        if label == uni_label:
            etn_p.append(etn_sample)
            dfan_p.append(dfan_sample)
            transzero_p.append(transzero_sample)
        else:
            continue
    etn_p, dfan_p, transzero_p = torch.stack(etn_p, dim=0).mean(0), torch.stack(dfan_p, dim=0).mean(0), torch.stack(transzero_p, dim=0).mean(0)
    etn_p, dfan_p, transzero_p = F.softmax(etn_p, dim=0), F.softmax(dfan_p, dim=0), F.softmax(transzero_p, dim=0)
    etn_p, dfan_p, transzero_p = etn_p.numpy(), dfan_p.numpy(), transzero_p.numpy()
    gt = torch.from_numpy(attrs_mat[uni_label])
    gt = F.softmax(gt, dim=-1).detach().numpy()
    kl_etn = np.sum(kl_div(gt, etn_p))
    kl_dfan = np.sum(kl_div(gt, dfan_p))
    kl_transzero = np.sum(kl_div(gt, transzero_p))
    kl_e.append(kl_etn)
    kl_d.append(kl_dfan)
    kl_t.append(kl_transzero)
    print(f'Label{uni_label}, ETN KL:{kl_etn}, DFAN KL:{kl_dfan}, TransZero KL:{kl_transzero}')

import matplotlib.pyplot as plt
import numpy as np

unique_labels = np.array(unique_labels)
x = np.arange(len(unique_labels))
width = 0.3
fig, ax = plt.subplots(figsize=(12, 8))
methods = ['EAFL', 'DFAN', 'TransZero']
kl_values = [kl_e, kl_d, kl_t]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (method, kl, color) in enumerate(zip(methods, kl_values, colors)):
    ax.bar(x + i * width, kl, width, label=f'{method}', color=color)
ax.set_ylabel('KL Divergence', fontsize=12)
# ax.set_title('KL Divergence Comparison', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(class_name[unique_labels], rotation=90, fontsize=12)
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('KL/KL_Comparison.png')
plt.show()

