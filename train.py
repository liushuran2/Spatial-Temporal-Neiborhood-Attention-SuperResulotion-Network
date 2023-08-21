import argparse
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from tqdm import tqdm
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Input a config file.')
parser.add_argument('--config', help='Config file path')
args = parser.parse_args()
f = open(args.config)
config = yaml.load(f, Loader=yaml.FullLoader)

import numpy as np
from torch.utils.data import DataLoader
from TrainDataset import TrainDataset
from TestDataset import TestDataset
import models
import torch
import loss

batchsize = config['batchsize']
scale = config['scale']

os.makedirs(config['checkpoint_folder'], exist_ok=True)
model = torch.nn.DataParallel(models.mana(config,is_training=True)).cuda()

if config['hot_start']:
    checkpt=torch.load(config['hot_start_checkpt'])
    model.module.load_state_dict(checkpt)

loss_fn = loss.lossfun()

writer = SummaryWriter(log_dir=config['checkpoint_folder'])

train_dataset = TrainDataset(config['dataset_path'], patch_size=config['patch_size'], scale=scale)
dataloader = DataLoader(dataset=train_dataset,
                        batch_size=batchsize,
                        shuffle=True,
                        num_workers=config['num_workers'],
                        pin_memory=True)
test_dataset = TestDataset(config['valid_dataset_path'])
test_dataloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config['num_workers'],
                        pin_memory=True)

def calc_psnr(sr, hr, scale=3, rgb_range=255, dataset=None):
    diff = (sr - hr)
    diff = diff.cpu().detach().numpy()
    mse = np.mean((diff) ** 2)
    return -10 * math.log10(mse)

count = 0
stage1=config['stage1']
stage2=config['stage2']
stage3=config['stage3']
best_valid = 100
for epoch in range(0, config['epoch']):
    loss_list=[]
    loss_list2=[]
    test_list=[]
    psnr_list=[]
    qloss_list=[]
    valid_list = []
    model.train()
    with tqdm(dataloader, desc="Training Model") as tepoch:
        for inp, gt in tepoch:
            tepoch.set_description(f"Training Model--Epoch {epoch}")
            if count==0:
                for p in model.module.parameters():
                    p.requires_grad=True
                for p in model.module.temporal_spatial.W_z1.parameters():
                    p.requires_grad=False
                model.module.temporal_spatial.mb.requires_grad=False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=5e-5, betas=(0.5, 0.999))

            elif count==stage1:
                for p in model.module.parameters():
                    p.requires_grad=False
                for p in model.module.temporal_spatial.W_z1.parameters():
                    p.requires_grad=True
                model.module.temporal_spatial.mb.requires_grad=True
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=2e-4, betas=(0.5, 0.999))

            elif count==stage2:
                for p in model.module.parameters():
                    p.requires_grad=True
                model.module.temporal_spatial.mb.requires_grad=True
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=2e-5, betas=(0.5, 0.999))

            elif count==stage3:
                for p in model.module.parameters():
                    p.requires_grad=True
                model.module.temporal_spatial.mb.requires_grad=True
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=2e-5, betas=(0.5, 0.999))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1)
    
    
            inp = inp.float().cuda()
            gt = gt.float().cuda()
            
            optimizer.zero_grad()
            oup,qloss = model(inp)
            
            if count<stage1:
                loss,l1loss = loss_fn(gt, oup)
                loss = loss.mean()
                loss.backward()
                loss_list.append(loss.data.cpu())
                loss_list2.append(l1loss.data.cpu())
                optimizer.step()
                tepoch.set_postfix({'L1 Loss': loss.data.cpu().numpy(),'L2 Loss': l1loss.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage1'})
                
                    
            elif count<stage2:
                loss=torch.mean(qloss)
                loss.backward()
                qloss_list.append(loss.data.cpu())
                loss1,l1loss = loss_fn(gt, oup)
                loss1 = loss1.mean()
                loss_list.append(loss1.data.cpu())
                loss_list2.append(l1loss.data.cpu())
                optimizer.step()
                tepoch.set_postfix({'Quantize Loss:': loss1.data.cpu().numpy(),'L2 Loss': l1loss.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage2'})
                
            else:
                loss,l1loss = loss_fn(gt, oup)
                loss = loss.mean()
                loss.backward()
                loss_list.append(loss.data.cpu())
                loss_list2.append(l1loss.data.cpu())
                optimizer.step()
                tepoch.set_postfix({'L1 Loss': loss.data.cpu().numpy(),'L2 Loss': l1loss.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage3'})

            count += 1
    
            if count % config['N_save_checkpt'] == 0:
                tepoch.set_description("Training Model--Saving Checkpoint")
                torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + config['checkpoint_name'])
                with torch.no_grad():
                    model.eval()
                    with tqdm(test_dataloader, desc="Valid Model") as tepoch:
                        for inp, gt in tepoch:
                            model.eval()
                            inp = inp.float().cuda()
                            gt = gt.float().cuda()
                            optimizer.zero_grad()
                            oup,qloss = model(inp)
                            loss,l1loss = loss_fn(gt, oup[:,:,:,:])
                            loss = loss.mean()
                            valid_list.append(l1loss.data.cpu())
                writer.add_scalar('Valid/loss', torch.mean(torch.stack(valid_list)), count / config['N_save_checkpt'])
                if torch.mean(torch.stack(valid_list)) < best_valid:
                    best_valid = torch.mean(torch.stack(valid_list))
                    torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + 'checkptbest.pt')
                
    writer.add_scalar('Train/loss', torch.mean(torch.stack(loss_list)), epoch)
    if count < stage2 and count > stage1:
        writer.add_scalar('Train/qloss', torch.mean(torch.stack(qloss_list)), epoch)
    writer.add_scalar('Train/l1loss', torch.mean(torch.stack(loss_list2)), epoch)
    if count > stage3:
        scheduler.step()
        writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], epoch)
writer.close()