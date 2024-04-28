from __future__ import print_function
import yaml
import os
import easydict
import numpy as np
import torch.optim as optim
import torch.nn.functional as func
from torch.autograd import Variable
import torchvision.transforms as transforms
from data_loader.get_loader import get_loader
from utils.lr_schedule import inv_lr_scheduler
from utils.loss import *
from utils.utils import *
from datetime import datetime
from easydl import *

### Training settings
import argparse
parser = argparse.ArgumentParser(description='Pytorch STUN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--setting', default='unida', 
                    type=str, 
                    choices=['unida', 'oda'], 
                    help='type of domain adaptation setting')
parser.add_argument('--dataset', default='office', 
                    type=str, 
                    choices=['office', 'officehome','visda'], 
                    help='type of dataset')
parser.add_argument('--source_domain', 
                    type=str)
parser.add_argument('--target_domain', 
                    type=str)
parser.add_argument('--source_path', 
                    type=str, 
                    default='./utils/source_list.txt', 
                    help='path to source domain list')
parser.add_argument('--target_path', 
                    type=str, 
                    default='./utils/target_list.txt', 
                    help='path to target domain list')
parser.add_argument("--seed", 
                    type=int, 
                    default=42, 
                    help='seed to control the randomness')
parser.add_argument("--gpu_devices", 
                    type=int, 
                    nargs='+', 
                    default=None, 
                    help="")
parser.add_argument('--num_classifiers', 
                    default=10, 
                    type=int,
                    help='number of sampled stochastic binary networks')
parser.add_argument('--conf_threshold', 
                    type=float, 
                    default=0.5,
                    help='threshold to remove underconfident predictions')
parser.add_argument('--lambda_open_entropy', 
                    type=float, 
                    default=0.5,
                    help='open-set entropy loss scaling factor')
parser.add_argument('--lambda_consis', 
                    type=float, 
                    default=0.1,
                    help='consistency regularization loss factor')
parser.add_argument('--lambda_adversarial', 
                    type=float, 
                    default=0.1,
                    help='adversarial learning loss scaling factor')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
set_seed(args.seed)
args.cuda = torch.cuda.is_available()


def train(source,target,config):
    conf = yaml.load(open(config),  Loader=yaml.FullLoader)
    conf = easydict.EasyDict(conf)
    source_data = source
    target_data = target
    evaluation_data = target
    batch_size = conf.data.dataloader.batch_size
    n_share = conf.data.dataset.n_share
    n_source_private = conf.data.dataset.n_source_private
    num_class = n_share + n_source_private

    ### Data transformations
    data_transforms = {
    source_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    target_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    evaluation_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    ### Data loading
    source_loader, target_loader, \
    test_loader, target_folder = get_loader(source_data, target_data, 
                                            evaluation_data, data_transforms, 
                                            batch_size=batch_size, return_id=True,
                                            balanced=conf.data.dataloader.class_balance)
  
    
    ### Networks initialization
    F, dim = get_model_mme(conf.model.base_model, num_class=num_class) # Feature extractor (ResNet-50)
    SBN = StochasticClassifier(dim,1,2*num_class) # Stochastic Binary Network
    D = AdversarialNetwork(dim) # Adversarial domain discriminator
    device = torch.device("cuda")
    F.to(device)
    SBN.to(device)
    D.to(device)
 
    params = []
    for key, value in dict(F.named_parameters()).items():
        if 'bias' in key:
            params += [{'params': [value], 'lr': conf.train.multi,
                        'weight_decay': conf.train.weight_decay}]
        else:
            params += [{'params': [value], 'lr': conf.train.multi,
                        'weight_decay': conf.train.weight_decay}]

    ### Optimizers initialization for networks     
    opt_f = optim.SGD(params, momentum=conf.train.sgd_momentum,
                    weight_decay=0.0005, nesterov=True)
    opt_sbn = optim.SGD(SBN.parameters(), lr=1.0,
                    momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                    nesterov=True)
    opt_d = optim.SGD(D.parameters(), lr=1.0, weight_decay=0.0005, momentum=conf.train.sgd_momentum, nesterov=True)

    F = nn.DataParallel(F)
    SBN = nn.DataParallel(SBN)
    D = nn.DataParallel(D)

    param_lr_f = []
    for param_group in opt_f.param_groups:
        param_lr_f.append(param_group["lr"])
    param_lr_sbn = []
    for param_group in opt_sbn.param_groups:
        param_lr_sbn.append(param_group["lr"])
    param_lr_d = []
    for param_group in opt_d.param_groups:
        param_lr_d.append(param_group["lr"])

    print('Training starts!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    for step in range(conf.train.min_step + 1):
        F.train()
        SBN.train()
        D.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_f, opt_f, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_sbn, opt_sbn, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_d,opt_d,step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step,power=0.75,gamma=10)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        img_t_bar = data_t[1]
        index_t = data_t[3]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        index_t = Variable(index_t.cuda())
        if len(img_t) < batch_size:
            break
        if len(img_s) < batch_size:
            break
        opt_f.zero_grad()
        opt_sbn.zero_grad()
        opt_d.zero_grad()
        
        ### One-vs-all loss for source data using hard negative classifier sampling
        feat_s = F(img_s)
        out_open = SBN(feat_s)
        out_open = out_open.view(out_open.size(0), 2, -1)
        out_open_prob = torch.softmax(out_open,dim=1)
        open_loss_pos, open_loss_neg = ova_loss(out_open_prob, label_s)
        loss_ova = 0.5 * (open_loss_pos + open_loss_neg)
        total_loss = loss_ova
        log_string = 'Train {}/{} \t ' \
                     'One-vs-all loss: {:.6f} ' 
        log_values = [step, conf.train.min_step, loss_ova.item()]

        ### Confidence score calculation for source data 
        prob_s = SBN(feat_s.detach())
        prob_s = prob_s.view(prob_s.size(0),2,-1)
        prob_s = torch.softmax(prob_s,dim=1)
        max_probs_src = prob_s.clone()

        for _ in range(args.num_classifiers-1): # Sampling m-1 stochastic binary networks (Step-1)
            prob_s = SBN(feat_s.detach())
            prob_s = prob_s.view(prob_s.size(0),2,-1)
            prob_s = torch.softmax(prob_s,dim=1) 
            max_probs_src += prob_s  # Calculating summation of outputs (Step-2)

        max_probs_src /= args.num_classifiers # Calculating average (Step-3)
        max_probs_src, _ = torch.max(max_probs_src[:,1,:], dim=1) # Maximum inlier prob. calculation (Step-4)
        max_probs_src = max_probs_src.detach()
            
        ### Open-set entropy for target data
        feat_t = F(img_t)
        out_open_target = SBN(feat_t)
        out_open_target = out_open_target.view(out_open_target.size(0), 2, -1)
        out_open_target_prob = torch.softmax(out_open_target,dim=1)
        ent_open = open_entropy(out_open_target_prob)
        total_loss += args.lambda_open_entropy * ent_open 
        log_values.append(ent_open.item())
        log_string += " Open-set entropy: {:.6f}"

        ### Predictions for strongly augmented target data
        feat_t_bar = F(img_t_bar)
        out_open_target_bar = SBN(feat_t_bar)
        out_open_target_bar = out_open_target_bar.view(out_open_target_bar.size(0), 2, -1)
        out_open_target_bar_prob = torch.softmax(out_open_target_bar,dim=1)
        
        ### Confidence score calculation for target data
        prob_t = SBN(feat_t.detach())
        prob_t = prob_t.view(prob_t.size(0),2,-1)
        prob_t = torch.softmax(prob_t,dim=1)
        max_probs_target = prob_t.clone()

        for _ in range(args.num_classifiers-1): # Sampling m-1 stochastic binary networks (Step-1)
            prob_t = SBN(feat_t.detach())
            prob_t = prob_t.view(prob_t.size(0),2,-1)
            prob_t = torch.softmax(prob_t,dim=1)
            max_probs_target += prob_t # Calculating summation of outputs (Step-2)

        max_probs_target /= args.num_classifiers # Calculating average (Step-3)
        max_probs_target, _ = torch.max(max_probs_target[:,1,:], dim=1) # Maximum inlier prob. calculation (Step-4)
        max_probs_target = max_probs_target.detach()

        mask = max_probs_target >= args.conf_threshold
      
        ### Consistency regularization loss calculation using deep discriminative clustering
        tensor_list = out_open_target_prob[mask]
        targets_u_aux = out_open_target_prob[mask] / tensor_list.sum(0, keepdim=True).pow(0.5) 
        targets_u_aux /= targets_u_aux.sum(1, keepdim=True)
        loss_consis = open_cross_entropy(targets_u_aux,out_open_target_bar_prob[mask]) 
        total_loss += args.lambda_consis*loss_consis
        log_values.append(loss_consis.item())
        log_string += " Consistency regularization loss: {:.6f}"

        ### Adversarial loss calculation
        domain_prob_discriminator_source = D(feat_s)
        domain_prob_discriminator_target  = D(feat_t)
        max_probs_src = max_probs_src.resize(max_probs_src.size()[0],1)
        max_probs_target = max_probs_target.resize(max_probs_target.size()[0],1)

        tmp = max_probs_src * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        loss_adv = torch.mean(tmp, dim=0, keepdim=True)
        tmp =  max_probs_target * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
        loss_adv  += torch.mean(tmp, dim=0, keepdim=True)
     
        loss_adv  = args.lambda_adversarial * loss_adv 
        loss_adv  = torch.squeeze(loss_adv)
        log_values.append(loss_adv)
        log_string += " Adversarial loss: {:.6f} "
        total_loss += loss_adv 

        total_loss.backward()
        opt_f.step()
        opt_sbn.step()
        opt_d.step()
        opt_f.zero_grad()
        opt_sbn.zero_grad()
        opt_d.zero_grad()

        if step % conf.train.log_interval == 0:
            print(log_string.format(*log_values))
        
        if step > 0 and step % conf.test.test_interval == 0:
            D.eval()
            acc_all, acc_known, acc_unknown, h_score = test(step, test_loader, n_share, F,
                                   SBN, open_class = num_class)
            print("step {:.2f}: acc all {:.2f}  acc known {:.2f} acc unknown {:.2f} h_score {:.2f}".format(step,acc_all,acc_known,acc_unknown,h_score))
            F.train()
            SBN.train()
            D.train()
    return h_score, acc_known, acc_unknown
            
def test(step, dataset_test, n_share, F, SBN,
         open_class = None):
    F.eval()
    SBN.eval()
    ## Known Score Calculation.
    correct = 0 # holds total number of correct predictions including predictions on target private data
    size = 0 # holds total number of samples
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)] # contain values from 0 to |L_s - 1|
    class_list.append(open_class)
    
    for _, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            feat = F(img_t)
            out_open = SBN(feat,mode='test')
            out_open_prob = func.softmax(out_open.view(out_open.size(0), 2, -1),1)

            prob_max , pred = torch.max(out_open_prob[:,1,:], dim=1)
            ind_unk = np.where(prob_max.data.cpu().numpy() < args.conf_threshold)[0]
            pred[ind_unk] = open_class # Assigns label of unknown classs (f.e, 200 for DomainNet) to the samples detected as unknown.

            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0])) # calculating number of correct predictions for each class
                per_class_num[i] += float(len(t_ind[0])) # calculating total number of samples for each class
            size += k

    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size) # Gives overall idea of global predictions of model
    known_acc = per_class_acc[:len(class_list)-1].mean() # Accuracy on known classes
    unknown = per_class_acc[-1] # Accuracy on unknown classes
    h_score = 2 * known_acc * unknown / (known_acc + unknown) # Simple H-score

    return acc_all, known_acc * 100, unknown * 100, h_score*100 

if args.setting == 'unida':

    if args.dataset == 'office':
        domains = ['amazon', 'webcam', 'dslr']
        avg = 0.0
        avg_acc_k = 0.0
        avg_acc_u = 0.0
        dic = {}

        for source in domains:
            for target in domains:
                if source == target:
                    continue
                start_time = datetime.now()
                source_  = f"./txt/source_{source}_opda.txt"
                target_ = f"./txt/target_{target}_opda.txt"
                config = f"configs/{args.dataset}-train-config_opda.yaml"

                h_score,acc_k,acc_u = train(source_,target_,config)
                dic[(source,target,"h_score")] = h_score
                dic[(source,target,"acc_k")] = acc_k
                dic[(source,target,"acc_u")] = acc_u

                avg += h_score
                avg_acc_k += acc_k
                avg_acc_u += acc_u
                end_time = datetime.now()
                duration = end_time - start_time
                print(f'{source} -> {target} :  {duration}')
        
        print()
        print('================================================================================')
        print('H-scores')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                h_score = dic[(source,target,"h_score")] 
                h_score = "{:.2f}".format(h_score)
                print(h_score,end = ', ')
        
        avg = avg/6
        avg = "{:.2f}".format(avg)
        print(avg)

        print()
        print('================================================================================')
        print('Accuracy known')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                acc_k = dic[(source,target,"acc_k")] 
                acc_k = "{:.2f}".format(acc_k)
                print(acc_k,end = ', ')
        
        avg_acc_k = avg_acc_k/6
        avg_acc_k = "{:.2f}".format(avg_acc_k)
        print(avg_acc_k)

        print()
        print('================================================================================')
        print('Accuracy unknown')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                acc_u = dic[(source,target,"acc_u")] 
                acc_u = "{:.2f}".format(acc_u)
                print(acc_u,end = ', ')
        
        avg_acc_u = avg_acc_u/6
        avg_acc_u = "{:.2f}".format(avg_acc_u)
        print(avg_acc_u)

    elif args.dataset == 'officehome':
        domains = ['Art', 'Clipart', 'Product', 'Real']
        avg = 0.0
        avg_acc_k = 0.0
        avg_acc_u = 0.0
        dic = {}

        for source in domains:
            for target in domains:
                if source == target:
                    continue
                start_time = datetime.now()
                source_  = f"./txt/source_{source}_opda.txt"
                target_ = f"./txt/target_{target}_opda.txt"
                config = f"configs/{args.dataset}-train-config_opda.yaml"

                h_score,acc_k,acc_u = train(source_,target_,config)
                dic[(source,target,"h_score")] = h_score
                dic[(source,target,"acc_k")] = acc_k
                dic[(source,target,"acc_u")] = acc_u

                avg += h_score
                avg_acc_k += acc_k
                avg_acc_u += acc_u
                end_time = datetime.now()
                duration = end_time - start_time
                print(f'{source} -> {target} :  {duration}')
        
        print()
        print('================================================================================')
        print('H-scores')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                h_score = dic[(source,target,"h_score")] 
                h_score = "{:.2f}".format(h_score)
                print(h_score,end = ', ')
        
        avg = avg/12
        avg = "{:.2f}".format(avg)
        print(avg)

        print()
        print('================================================================================')
        print('Accuracy known')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                acc_k = dic[(source,target,"acc_k")] 
                acc_k = "{:.2f}".format(acc_k)
                print(acc_k,end = ', ')
        
        avg_acc_k = avg_acc_k/12
        avg_acc_k = "{:.2f}".format(avg_acc_k)
        print(avg_acc_k)

        print()
        print('================================================================================')
        print('Accuracy unknown')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                acc_u = dic[(source,target,"acc_u")] 
                acc_u = "{:.2f}".format(acc_u)
                print(acc_u,end = ', ')
        
        avg_acc_u = avg_acc_u/12
        avg_acc_u = "{:.2f}".format(avg_acc_u)
        print(avg_acc_u)

    elif args.dataset == "visda":
        start_time = datetime.now()
        source_  = f"./txt/source_visda_opda.txt"
        target_ = f"./txt/target_visda_opda.txt"
        config = f"configs/{args.dataset}-train-config_opda.yaml"
        h_score,acc_k,acc_u = train(source_,target_,config)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f' Syn -> Real :  {duration}')

    else:
        raise ValueError("Wrong dataset!")


elif args.setting == 'oda':

    if args.dataset == 'office':
        domains = ['amazon', 'webcam', 'dslr']
        avg = 0.0
        avg_acc_k = 0.0
        avg_acc_u = 0.0
        dic = {}

        for source in domains:
            for target in domains:
                if source == target:
                    continue
                start_time = datetime.now()
                source_  = f"./txt/source_{source}_obda.txt"
                target_ = f"./txt/target_{target}_obda.txt"
                config = f"configs/{args.dataset}-train-config_ODA.yaml"

                h_score,acc_k,acc_u = train(source_,target_,config)
                dic[(source,target,"h_score")] = h_score
                dic[(source,target,"acc_k")] = acc_k
                dic[(source,target,"acc_u")] = acc_u

                avg += h_score
                avg_acc_k += acc_k
                avg_acc_u += acc_u
                end_time = datetime.now()
                duration = end_time - start_time
                print(f'{source} -> {target} :  {duration}')
        
        print()
        print('================================================================================')
        print('H-scores')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                h_score = dic[(source,target,"h_score")] 
                h_score = "{:.2f}".format(h_score)
                print(h_score,end = ', ')
        
        avg = avg/6
        avg = "{:.2f}".format(avg)
        print(avg)

        print()
        print('================================================================================')
        print('Accuracy known')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                acc_k = dic[(source,target,"acc_k")] 
                acc_k = "{:.2f}".format(acc_k)
                print(acc_k,end = ', ')
        
        avg_acc_k = avg_acc_k/6
        avg_acc_k = "{:.2f}".format(avg_acc_k)
        print(avg_acc_k)

        print()
        print('================================================================================')
        print('Accuracy unknown')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                acc_u = dic[(source,target,"acc_u")] 
                acc_u = "{:.2f}".format(acc_u)
                print(acc_u,end = ', ')
        
        avg_acc_u = avg_acc_u/6
        avg_acc_u = "{:.2f}".format(avg_acc_u)
        print(avg_acc_u)

    elif args.dataset == 'officehome':
        domains = ['Art', 'Clipart', 'Product', 'Real']
        avg = 0.0
        avg_acc_k = 0.0
        avg_acc_u = 0.0
        dic = {}

        for source in domains:
            for target in domains:
                if source == target:
                    continue
                start_time = datetime.now()
                source_  = f"./txt/source_{source}_obda.txt"
                target_ = f"./txt/target_{target}_obda.txt"
                config = f"configs/{args.dataset}-train-config_ODA.yaml"

                h_score,acc_k,acc_u = train(source_,target_,config)
                dic[(source,target,"h_score")] = h_score
                dic[(source,target,"acc_k")] = acc_k
                dic[(source,target,"acc_u")] = acc_u

                avg += h_score
                avg_acc_k += acc_k
                avg_acc_u += acc_u
                end_time = datetime.now()
                duration = end_time - start_time
                print(f'{source} -> {target} :  {duration}')
        
        print()
        print('================================================================================')
        print('H-scores')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                h_score = dic[(source,target,"h_score")] 
                h_score = "{:.2f}".format(h_score)
                print(h_score,end = ', ')
        
        avg = avg/12
        avg = "{:.2f}".format(avg)
        print(avg)

        print()
        print('================================================================================')
        print('Accuracy known')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                acc_k = dic[(source,target,"acc_k")] 
                acc_k = "{:.2f}".format(acc_k)
                print(acc_k,end = ', ')
        
        avg_acc_k = avg_acc_k/12
        avg_acc_k = "{:.2f}".format(avg_acc_k)
        print(avg_acc_k)

        print()
        print('================================================================================')
        print('Accuracy unknown')
        print('================================================================================')
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                acc_u = dic[(source,target,"acc_u")] 
                acc_u = "{:.2f}".format(acc_u)
                print(acc_u,end = ', ')
        
        avg_acc_u = avg_acc_u/12
        avg_acc_u = "{:.2f}".format(avg_acc_u)
        print(avg_acc_u)

    elif args.dataset == 'visda':
        start_time = datetime.now()
        source_  = f"./txt/source_visda_obda.txt"
        target_ = f"./txt/target_visda_obda.txt"
        config = f"configs/{args.dataset}-train-config_ODA.yaml"
        h_score,acc_k,acc_u = train(source_,target_,config)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f' Syn -> Real :  {duration}')

    else:
        raise ValueError("Wrong dataset!")

else:
    ValueError("Wrong setting!")