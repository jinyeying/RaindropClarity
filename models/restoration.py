import torch
import torch.nn as nn
import utils
import torchvision
import os
from torchvision.transforms.functional import crop

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion


        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='Raindrop_DiT', r=None, sid = None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                #print(x.shape,y)
                if sid:
                    y = y[0]
                    if sid+'__' in y:
                        # print(self.args.image_folder, self.config.data.dataset)
                        # print(i, x.shape, y)
                        datasetname =  y.split('__')[0]
                        id = y.split('__')[1]
                        frame = y.split('__')[2]
                        print(datasetname, id, frame)
                        x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x

                        x_cond = x[:, :3, :, :].to(self.diffusion.device)
                        x_gt = x[:, 3:, :, :].to(self.diffusion.device)
                        #print('x_cond = ',x_cond.shape)

                        utils.logging.save_image(x_cond[:, :, :, :], os.path.join(image_folder, datasetname,'input',sid, f"{frame}.png"))
                        utils.logging.save_image(x_gt[:, :, :, :], os.path.join(image_folder,datasetname, 'gt', sid, f"{frame}.png"))

                        if self.args.test_set in ['RDiffusion', 'Raindrop_DiT']:
                            x_output = self.diffusive_restoration(x_cond, r=r)
                        else:
                            input_res = self.config.data.image_size
                            print('input_res',input_res)
                            stride = 16

                            h_list = [i for i in range(0, x_cond.shape[2] - input_res + 1, stride)]
                            w_list = [i for i in range(0, x_cond.shape[3] - input_res + 1, stride)]
                            h_list = h_list + [x_cond.shape[2]-input_res]
                            w_list = w_list + [x_cond.shape[3]-input_res]

                            corners = [(i, j) for i in h_list for j in w_list]
                            #print('-corners-',corners)

                            p_size = input_res
                            x_grid_mask = torch.zeros_like(x_cond).cuda()
                            for (hi, wi) in corners:
                                x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

                            et_output = torch.zeros_like(x_cond).cuda()

                            manual_batching_size = 32
                            x_cond_patch = torch.cat([crop(x_cond, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                            for i in range(0, len(corners), manual_batching_size):
                                # print('-x_cond_patch[i:i+manual_batching_size]-',x_cond_patch[i:i+manual_batching_size].shape)
                                
                                if self.args.test_set == 'atgan':
                                    _,_,_,Output = self.diffusion.model( data_transform(x_cond_patch[i:i+manual_batching_size]).float() )
                                else:
                                    Output = self.diffusion.model( data_transform(x_cond_patch[i:i+manual_batching_size]).float() )

                                for didx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += Output[didx]

                            x_output = torch.div(et_output, x_grid_mask)

                        x_output = inverse_data_transform(x_output)
                        output_image_path = os.path.join(image_folder, datasetname, 'output', sid, f"{frame}.png")
                        utils.logging.save_image(x_output, output_image_path)
                        print(f"Output image saved at: {output_image_path}")

    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        #print('p_size = ',p_size)
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        #print('h_list = ',h_list)
        #print('w_list = ',w_list)
        corners = [(i, j) for i in h_list for j in w_list]
        # print('corners = ',corners)
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        #print('x = ', x.shape)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]

        if h%16 !=0:
            h_list = h_list +[h - output_size]
        if w%16 !=0:
            w_list = w_list +[w - output_size]
        return h_list, w_list
