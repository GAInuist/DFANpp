import pickle
import scipy.io as sio
from torch import optim
from sklearn.model_selection import train_test_split
from network.ETN import ETN
from Loss_dict import compute_cosine_loss, get_semantic_loss, compute_reg_loss
import matplotlib.pyplot as plt
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import warnings
from Model_Transfer import *
from utils import *
from config import *
from data import *

warnings.filterwarnings('ignore')

args = get_config_parser().parse_args()
for k, v in sorted(vars(args).items()):
    print(k, '=', v)

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

# tensorboard
log_dir = f'./runs/{args.DATASET}/{args.pref}'
writer = SummaryWriter(log_dir=log_dir)

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

trainTransform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

testTransform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])


def train(model, data_loader, train_attrbs, optimizer, attr_mat, args):
    # initialize variables to monitor training and validation loss
    """ begin the model's training """
    model.train()
    tk = tqdm(data_loader)
    lamb_1, lamb_2, lamb_3 = args.lambda_
    for batch_idx, (data, label) in enumerate(tk):
        # get data label and attribute
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        data_attribute = torch.from_numpy(attr_mat).cuda()[label]

        result_dict = model(data, w2v)
        attr_coarse = result_dict['F_coar']
        local_result, global_result = result_dict['local_result'], result_dict['global_result']
        local_bias, global_bias = result_dict['local_bias'], result_dict['global_bias']

        # cosine loss for attribute_feature init
        loss_cosine = compute_cosine_loss(attr_coarse)
        # get reg loss for APP's Attn Mask
        loss_reg = compute_reg_loss(result_dict, class_prototype=data_attribute, bias_local=local_bias)
        # get consistency loss for local and global
        loss_cons = get_semantic_loss(result_dict, local_bias, global_bias, data_attribute)
        # logit_loss compute
        local_result = local_result + local_bias
        global_result = global_result + global_bias
        logit_g = global_result @ train_attrbs.T
        logit_l = local_result @ train_attrbs.T
        loss_global = F.cross_entropy(logit_g, label)
        loss_local = F.cross_entropy(logit_l, label)

        loss = loss_local + loss_global + lamb_1 * loss_cosine + lamb_2 * loss_reg + lamb_3 * loss_cons
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), label.shape[0])
        loss_local_meter.update(loss_local.item(), label.shape[0])
        loss_global_meter.update(loss_global.item(), label.shape[0])
        loss_reg_meter.update(lamb_2 * loss_reg.item(), label.shape[0])
        loss_cosine_meter.update(loss_cosine.item(), label.shape[0])
        loss_cons_meter.update(lamb_3 * loss_cons.item(), label.shape[0])
        tk.set_postfix({"loss": loss_meter.avg,
                        "loss_global": loss_global_meter.avg,
                        "loss_local": loss_local_meter.avg,
                        "loss_cosine": loss_cosine_meter.avg,
                        "loss_reg": loss_reg_meter.avg,
                        "loss_cons": loss_cons_meter.avg})
    # print training/validation statistics
    print('Train: Average loss: {:.4f}\n'.format(loss_meter.avg))


model = ETN(dim=args.v_embedding, attr_num=args.attr_num, drop_rate=args.drop_rate, n_head=args.n_head, word_vector_length=args.word_embedding_length).cuda()

############# with pretraining #############
if args.pretrain_path is not None:
    pth_dict = torch.load(args.pretrain_path, map_location='cuda:0')
    if args.is_train:
        model_dict = model_transfer(model, pth_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(pth_dict)
        
if args.is_train or args.cs:
    for param in model.parameters():
        param.requires_grad = True
        optimizer = torch.optim.AdamW(
            [{"params": model.v_encoder.parameters(), "lr": 0.000001, "weight_decay": 0.001},
             {"params": model.coarse_extractor.parameters(), "lr": 0.0001, "weight_decay": 0.00001},
             {"params": model.W1.parameters(), "lr": 0.001, "weight_decay": 0.00001},
             {"params": model.W_g.parameters(), "lr": 0.001, "weight_decay": 0.00001},
             {"params": model.W_l.parameters(), "lr": 0.001, "weight_decay": 0.00001},
             {"params": model.CDM.parameters(), "lr": 0.0001, "weight_decay": 0.00001},
             {"params": model.VBL.parameters(), "lr": 0.0001, "weight_decay": 0.001},
             {"params": model.APP.parameters(), "lr": 0.0001, "weight_decay": 0.00001},
             {"params": model.norm.parameters(), "lr": 0.0001, "weight_decay": 0.00001},
             {"params": model.GAFR.parameters(), "lr": 0.0001, "weight_decay": 0.00001},
             {"params": model.CPP.parameters(), "lr": 0.0001, "weight_decay": 0.001}])

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 100, 150], gamma=0.6)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

# train attributes
train_attrbs = attrs_mat[uniq_train_labels]
train_attrbs_tensor = torch.from_numpy(train_attrbs).cuda()
# trainval attributes
trainval_attrbs = attrs_mat[uniq_trainval_labels]
trainval_attrbs_tensor = torch.from_numpy(trainval_attrbs).cuda()

loss_meter = AverageMeter()
loss_global_meter, loss_local_meter, loss_cosine_meter = AverageMeter(), AverageMeter(), AverageMeter()
loss_reg_meter, loss_cons_meter = AverageMeter(), AverageMeter()

### =======================================================train========================================================
if __name__ == '__main__':
    if args.is_train:
        trainval_data_loader = get_loader(args, ROOT, trainval_files, trainval_labels_based0, trainTransform,
                                          is_sample=True,
                                          count_labels=counts_trainval_labels)
        test_seen_data_loader = get_loader(args, ROOT, test_seen_files, test_seen_labels, testTransform,
                                           is_sample=False)
        test_unseen_data_loader = get_loader(args, ROOT, test_unseen_files, test_unseen_labels, testTransform,
                                             is_sample=False)
        harmonic_best = []
        logging.basicConfig(level=logging.INFO, filename='Training_Log_DFANpp_lamb1-0.01_lamb2-0.01.log', format='%(message)s')
        for epoch in range(1, args.Num_Epochs):
            print('Train Val Epoch: ', epoch)
            train(model, trainval_data_loader, trainval_attrbs_tensor, optimizer, attrs_mat, args)
            lr_scheduler.step()
            test_gamma_best, best_test_dict = [], {}
            gamma_list = [0.9, 0.95]
            for gamma in tqdm(gamma_list, desc="Processing gamma values"):
                metric_dict = test_GZSL(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader,
                                        test_unseen_labels, attrs_mat, gamma, w2v, beta1=args.beta1, beta2=args.beta2)
                test_gamma_best.append(metric_dict['H'])
                if metric_dict['H'] >= max(test_gamma_best):
                    best_test_dict = metric_dict
            H, gzsl_seen_acc, gzsl_unseen_acc, best_gamma = (best_test_dict['H'], best_test_dict['gzsl_seen'],
                                                             best_test_dict['gzsl_unseen'], best_test_dict['gamma'])
            H_local, local_seen_acc, local_unseen_acc = (best_test_dict['H_local'], best_test_dict['local_seen'], best_test_dict['local_unseen'])
            H_global, global_seen_acc, global_unseen_acc = (best_test_dict['H_global'], best_test_dict['global_seen'], best_test_dict['global_unseen'])
            print('GZSL Seen: averaged per-class accuracy: {0:.2f}, Local Seen: {1:.2f}, Global Seen: {2:.2f}'.format(gzsl_seen_acc, local_seen_acc, global_seen_acc))
            print('GZSL Unseen: averaged per-class accuracy: {0:.2f}, Local Unseen: {1:.2f}, Global Unseen: {2:.2f}'.format(gzsl_unseen_acc, local_unseen_acc, global_unseen_acc))
            print('GZSL: harmonic mean (H): {0:.2f}, Local H: {1:.2f}, Global H: {2:.2f}'.format(H, H_local, H_global))
            print('GZSL: best_gamma: {0:.2f}'.format(best_gamma))

            harmonic_best.append(best_test_dict['H'])
            print(f"Now the best harmonic is {max(harmonic_best)}, index is {harmonic_best.index(max(harmonic_best)) + 1}")
            if best_test_dict['H'] >= max(harmonic_best):
                print(' .... Saving model ...')
                h = best_test_dict['H']
                save_path_rm = str(args.DATASET) + f'_DFANpp' + '.pth'
                ckpt_path = os.path.join('checkpoint', str(args.DATASET), f'{args.pref}')
                path = os.path.join(ckpt_path, save_path_rm)
                if not os.path.isdir(ckpt_path):
                    makedir(ckpt_path)
                torch.save(model.state_dict(), path)
                print(f"Now saving model with harmonic {h}")
            #### tensorboard
            writer.add_scalar('H', metric_dict['H'], epoch)
            writer.add_scalar('U', metric_dict['gzsl_unseen'], epoch)
            writer.add_scalar('S', metric_dict['gzsl_seen'], epoch)
            logging_info_flow(best_test_dict, epoch, loss_meter.avg)
        print(max(harmonic_best))

        
    beta1_values = np.linspace(0.0, 1.0, 21) 
    beta2_values = np.linspace(1.0, 0.0, 21)
    test_seen_data_loader = get_loader(args, ROOT, test_seen_files, test_seen_labels, testTransform,
                                       is_sample=False)
    test_unseen_data_loader = get_loader(args, ROOT, test_unseen_files, test_unseen_labels, testTransform,
                                         is_sample=False)
    for beta1, beta2 in zip(beta1_values, beta2_values):
        test_dict = test_GZSL(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader,
                                test_unseen_labels, attrs_mat, args.gamma, w2v, beta1=beta1, beta2=beta2)
        czsl, H, S, U = (test_dict['czsl'], test_dict['H'], test_dict['gzsl_seen'], test_dict['gzsl_unseen'])
        print(f"({beta1:.2f},{beta2:.2f}): czsl-{czsl}, H-{H}, S-{S}, U-{U}")
