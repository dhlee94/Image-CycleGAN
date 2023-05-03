import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import os
import time
import wandb
import pickle
import logging
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.utils import AverageMeter
from .criterion import *
from .optimizer import *
import copy

def train(generatorA=None, generatorB=None, discriminatorA=None, discriminatorB=None,
          write_iter_num=None, device=None, train_dataset=None, optimizerG=None, optimizerDA=None, optimizerDB=None, 
          criterion_G=nn.MSELoss(), criterion_C=nn.L1Loss(), criterion_P=nn.L1Loss(), lambda_i=None, lambda_c=None, file=None):

    scaler = torch.cuda.amp.GradScaler()
    assert train_dataset is not None, print("train_dataset is none")
    ave_accuracy = AverageMeter()
    #scaler = torch.cuda.amp.GradScaler()
    batches_done = 0
    generatorA = generatorA.to(device)
    generatorB = generatorB.to(device)
    discriminatorA = discriminatorA.to(device)
    discriminatorB = discriminatorB.to(device)
    generatorA.train()
    generatorB.train()
    discriminatorA.train()
    discriminatorB.train()
    for idx, (Image, Target) in enumerate(tqdm(train_dataset)):
        #model input data
        Image, Target = train_batch
        Image = Image.to(device, non_blocking=True)
        Target = Target.to(device, non_blocking=True)

        #Adversarial ground truths
        valid = Variable(Tensor(np.ones((Image.size(0), *discriminatorA.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((Target.size(0), *discriminatorA.output_shape))), requires_grad=False)
        optimizerG.zero_grad()
        #Identity Loss
        loss_id_A = criterion_P(generatorB(Image), Image)
        loss_id_B = criterion_P(generatorA(Target), Target)
        loss_identity = (loss_id_A+loss_id_B)/2

        #GAN Loss
        gen_A = generatorA(Image)
        loss_G_A = criterion_G(discriminatorA(gen_A), valid)
        gen_B = generatorB(Target)
        loss_G_B = criterion_G(discriminatorB(gen_B), valid)
        loss_G = (loss_G_A + loss_G_B)/2
        
        #Cycle Loss
        regen_B = generatorB(gen_A)
        loss_C_B = criterion_C(regen_B, Image)
        regen_A = generatorA(gen_B)
        loss_C_A = criterion_C(regen_A, Target)
        loss_C = (loss_C_B + loss_C_A)/2

        #Total Loss(Generator)
        loss_T = loss_G + lambda_c*loss_C + lambda_i*loss_identity
        loss_T.backward()
        optimizerG.step()

        #DiscriminatorA
        optimizerDA.zero_grad()
        loss_real = criterion_G(discriminatorA(Image), valid)
        loss_fake = criterion_G(discriminatorA(gen_A.detach()), fake)
        loss_DA = (loss_real + loss_fake)/2
        loss_DA.backward()
        optimizerDA.step()

        #DiscriminatorB
        optimizerDB.zero_grad()
        loss_real = criterion_G(discriminatorB(Target), valid)
        loss_fake = criterion_G(discriminatorB(gen_B.detach()), fake)
        loss_DB = (loss_real + loss_fake)/2
        loss_DB.backward()
        optimizerDB.step()

        if idx % write_iter_num == 0:
            tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(train_dataset)} '
                        f'Generator Loss : {loss_T :.4f} ' f'DiscriminatorA Loss : {loss_DA :.4f} '
                        f'DiscriminatorB Loss : {loss_DA :4f}')
        if idx % 2*write_iter_num == 0:
            tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(train_dataset)} '
                        f'Generator Loss : {loss_T :.4f} ' f'DiscriminatorA Loss : {loss_DA :.4f} '
                        f'DiscriminatorB Loss : {loss_DA :4f}', file=file)