import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
from models.transformer2d import My_DiT_test
from models.ICRA import create_gen_nets
from models.onego_genotypes_searched import architectures
from models.onego_train_model import Raincleaner_train
from models.IDT import create_IDT_nets
from models.Uformer import create_uformer_nets
from models.restormer import create_restormer_nets
from models.atgan import create_atgan_nets


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm

class AttentionLoss(nn.Module):
    def __init__(self, theta=0.8, iteration=4):
        super(AttentionLoss, self).__init__()
        self.theta = theta
        self.iteration = iteration
        self.loss = nn.MSELoss().cuda()

    def __call__(self, A_, M_):
        loss_ATT = None
        for i in range(1, self.iteration+1):
            if i == 1:
                loss_ATT = pow(self.theta, float(self.iteration-i)) * self.loss(A_[i-1],M_)
            else:
                loss_ATT += pow(self.theta, float(self.iteration-i)) * self.loss(A_[i-1],M_)
        return loss_ATT

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9996):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        if self.args.test_set == 'RDiffusion':
            self.model = DiffusionUNet(config)
            self.model_name = 'RDiffusion'
            assert self.config.data.image_size == 64, f"Expected image_size 64, but got {self.config.data.image_size}"
        
        if self.args.test_set == 'Raindrop_DiT':
            self.model = My_DiT_test(input_size=config.data.image_size)
            self.model_name = 'Raindrop_DiT'

        if self.args.test_set == 'onego':
            genotype = architectures['RD_V2']
            self.model = Raincleaner_train(genotype)
            self.model_name = 'onego'
            assert self.config.data.image_size == 64, f"Expected image_size 64, but got {self.config.data.image_size}"

        if self.args.test_set == 'ICRA256':
            self.model = create_gen_nets()
            self.model_name = 'ICRA256'
            assert self.config.data.image_size == 256, f"Expected image_size 256, but got {self.config.data.image_size}"

        if self.args.test_set == 'IDT':
            self.model = create_IDT_nets()
            self.model_name = 'IDT'
            assert self.config.data.image_size == 128, f"Expected image_size 128, but got {self.config.data.image_size}"

        if self.args.test_set == 'Uformer':
            self.model = create_uformer_nets()
            self.model_name = 'Uformer'
            assert self.config.data.image_size == 256, f"Expected image_size 256, but got {self.config.data.image_size}"

        if self.args.test_set == 'restormer':
            self.model = create_restormer_nets()
            self.model_name = 'restormer'
            assert self.config.data.image_size == 128, f"Expected image_size 128, but got {self.config.data.image_size}"

        if self.args.test_set == 'atgan':
            self.model = create_atgan_nets()
            self.model_name = 'atgan'
            assert self.config.data.image_size == 256, f"Expected image_size 256, but got {self.config.data.image_size}"
            self.criterionMSE = nn.MSELoss().cuda()
            self.criterionAtt = AttentionLoss(theta=0.8, iteration=4)

        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.args.test_set in ['RDiffusion', 'Raindrop_DiT']:
            self.ema_helper.load_state_dict(checkpoint['ema_helper'])
            if ema:
                self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))
    
    def get_mask(self, dg_img, img):
        # downgraded image - image
        mask = np.fabs(dg_img-img)
        # threshold under 30
        mask[np.where(mask<(30.0/255.0))] = 0.0
        mask[np.where(mask>0.0)] = 1.0
        #avg? max?
        # mask = np.average(mask, axis=2)
        mask = np.max(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return mask

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        LossL1 = torch.nn.L1Loss(reduce=True, size_average=True)

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                # print(i,x.shape,y)
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                if self.args.test_set in ['RDiffusion', 'Raindrop_DiT']:
                    x = x.to(self.device)
                    x = data_transform(x)

                    e = torch.randn_like(x[:, 3:, :, :])
                    b = self.betas

                    # antithetic sampling
                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    loss = noise_estimation_loss(self.model, x, t, e, b)

                elif self.args.test_set == 'atgan':
                    X_input = x[:, :3, :, :]
                    X_GT    = x[:, 3:, :, :]
                    M_ = []
                    for i in range(X_input.shape[0]):
                        M_.append(self.get_mask(X_input[i].numpy(),X_GT[i].numpy()))
                    M_ = np.array(M_)
                    M_ = torch.from_numpy(M_).cuda().float()
                    # print('-M_-',M_.shape)

                    X_input = X_input.to(self.device)
                    X_input = data_transform(X_input)
                    X_GT = X_GT.to(self.device)
                    X_GT = data_transform(X_GT)
                    A_, t1, t2, t3 = self.model(X_input)
                    S_ = [t1,t2,t3]
                    O_ = t3

                    loss1 = self.criterionMSE(O_,X_GT.detach())
                    loss2 = self.criterionAtt(A_,M_.detach())
                    # print('-X_output-',X_output.shape,X_input[:,:,midframe,:,:].shape)
                    loss = loss1 + loss2

                else:     
                    x = x.to(self.device)
                    x = data_transform(x)

                    X_input = x[:,:3,:,:]
                    X_GT    = x[:,3:,:,:]
                    X_output = self.model(X_input)
                    # print('-X_output-',X_output.shape,X_input[:,:,midframe,:,:].shape)
                    lossl1 = LossL1(X_output, X_GT)
                    loss = lossl1 

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.args.test_set in ['RDiffusion', 'Raindrop_DiT']:
                    self.ema_helper.update(self.model)
                data_start = time.time()

                if self.args.test_set in ['RDiffusion', 'Raindrop_DiT']:
                    if self.step % self.config.training.validation_freq == 0:
                        self.model.eval()
                        self.sample_validation_patches(val_loader, self.step)

                    if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                        checkpoint_path = os.path.join('Param/'+ self.config.data.dataset +'/' + self.model_name + '_ddpm')
                        utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                        }, filename=checkpoint_path)
                else:
                    if (epoch+1) % self.config.training.snapshot_freq  == 0:
                        checkpoint_path = os.path.join('Param/', self.config.data.dataset + '/' + self.model_name +'/'+'epoch'+str(epoch + 1))
                        utils.logging.save_checkpoint({
                            'epoch': epoch + 1,
                            'step': self.step,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'params': self.args,
                            'config': self.config
                        }, filename=checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
