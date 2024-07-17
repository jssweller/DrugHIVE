import os
import glob
import logging
from scipy.stats import pearsonr
import pytorch_lightning as pl
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adamax

from .blocks import GaussConvLayer
from .data import MolDatasetPDBBindZINCSplit, rot3d_random_safe, rot3d, trans3d, trans3d_random_safe
from .molecules import MolParser
from .models import AutoEncoderComplexSplit
from .trainutils import KLBalancer, recon_loss_fn, steric_loss_fn, batchnorm_loss_fn, spec_norm_loss_fn, average_distributed, filter_predictions, plot_scatterplot, get_gradients_norm


class HVAEComplexSplit(pl.LightningModule):
    '''Heirarchical Variational AutoEncoder with split resolution for inputs.'''

    def __init__(self, args, verbose=False):
        self.verbose = verbose
        super(HVAEComplexSplit, self).__init__()

        # this allows us to control optimizer.step in training_step()
        self.automatic_optimization = True

        # set some defaults
        args.input_grid_format2 = args.get('input_grid_format2', 'density')
        args.ouput_grid_format2 = args.get('output_grid_format2', 'density')

        args.n_channels_input = args.get('n_channels_input', 1)
        args.n_channels_input_prot = args.get('n_channels_input_prot', 1)
        args.n_channels_input2 = args.get(
            'n_channels_input2', len(args.data_args.channels_in_lig))
        args.n_channels_input_prot2 = args.get(
            'n_channels_input_prot2', len(args.data_args.channels_in_prot))

        args.decoder_noise = args.get('decoder_noise', 0)

        self.args = args

        self.logging = logging.getLogger("pytorch_lightning")
        self.hvae_model = AutoEncoderComplexSplit(args)
        self.gauss_conv_layer_lig = GaussConvLayer(var=args.data_args.gauss_var,
                                                   trunc=args.data_args.gauss_trunc,
                                                   resolution=args.data_args.resolution,
                                                   n_channels=args.n_channels_input2)

        self.gauss_conv_layer_prot = GaussConvLayer(var=args.data_args.gauss_var,
                                                    trunc=args.data_args.gauss_trunc,
                                                    resolution=args.data_args.resolution,
                                                    n_channels=args.n_channels_input_prot2)

        if self.args.output_grid_format == 'encoding':
            self.gauss_conv_layer_lig0 = GaussConvLayer(var=args.data_args.gauss_var,
                                                        trunc=args.data_args.gauss_trunc,
                                                        resolution=args.data_args.resolution,
                                                        n_channels=args.n_channels_input)

            self.gauss_conv_layer_sigma0 = GaussConvLayer(var=args.data_args.gauss_var,
                                                          trunc=args.data_args.gauss_trunc,
                                                          resolution=args.data_args.resolution,
                                                          n_channels=args.n_channels_input,
                                                          normalized=True
                                                          )

        self.hparams['args'] = args
        self.save_hyperparameters()
        self.configure_datasets()

        self.nelbo_vals = 0
        self.nelbo_count = 0
        self.nelbo_vals_valid = 0
        self.nelbo_count_valid = 0

        self.kl_balancer = KLBalancer(self.hvae_model.groups_per_res)

        if not hasattr(self.args, 'num_total_iter'):
            self.args.num_total_iter = len(
                self.dataset) // (args.batch_size * args.num_devices) * args.epochs
        if not hasattr(self.args, 'warmup_iters'):
            self.args.warmup_iters = len(
                self.dataset_train) // (args.batch_size * args.num_devices) * args.warmup_epochs

        self.skipped_updates = 0
        self.validation_step_outputs = [[], []]


    def training_step(self, batch, batch_idx):
        model = self.hvae_model
        args = self.args

        if batch_idx == 0:
            torch.cuda.set_device(self.device)
            self.kl_balancer.to_device(self.device)
        data = batch

        # create input densities
        if 'input_density_lig0' not in data.keys():
            data['input_density_lig'] = self.gauss_conv_layer_lig(
                data['input_encoding_lig'])
            data['target_density_lig'] = data['input_density_lig']

            data['input_density_prot'] = self.gauss_conv_layer_prot(
                data['input_encoding_prot'])
            data['target_density_prot'] = data['input_density_prot']

            if args.input_grid_format2 == 'density' or args.output_grid_format2 == 'density':
                data['input_density_lig2'] = torch.nn.functional.interpolate(
                    data['input_density_lig'], scale_factor=0.5, mode='trilinear')
                data['target_density_lig2'] = data['input_density_lig2']
                if args.input_grid_format2 == 'density':
                    data['input_density_prot2'] = torch.nn.functional.interpolate(
                        data['input_density_prot'], scale_factor=0.5, mode='trilinear')
                    data['target_density_prot2'] = data['input_density_prot2']

            # cumulative density/encoding
            element_channels_lig = self.dataset.grid_encoder_ligand.channels_elements_only
            element_channels_prot = self.dataset.grid_encoder_protein.channels_elements_only
            data['input_density_lig'] = data['input_density_lig0'] = data['input_density_lig'][:,
                                                                                               element_channels_lig].sum(dim=1, keepdim=True)
            data['target_density_lig'] = data['target_density_lig0'] = data['input_density_lig']
            data['input_density_prot'] = data['input_density_prot0'] = data['input_density_prot'][:,
                                                                                                  element_channels_prot].sum(dim=1, keepdim=True)
            data['target_density_prot'] = data['target_density_prot0'] = data['input_density_prot']

            target0 = data['target_density_lig0']
            target2 = data['target_density_lig2']

            data['input_encoding_lig'] = data['target_encoding_lig'] = data['input_encoding_lig'].cpu()
            data['input_encoding_prot'] = data['target_encoding_prot'] = data['input_encoding_prot'].cpu()

        noise_dec = 0.
        if args.decoder_noise > 0:
            noise_dec = min(1., self.global_step / args.warmup_iters) * \
                args.decoder_noise  # warm-up decoder noise

        (out_distp0, out_distp2), log_q, log_p, kl_all, kl_diag = model(
            data, noise_dec=noise_dec)

        output_dist = model.dist_from_output(out_distp0)
        output_dist2 = model.dist_from_output(out_distp2)

        self.kl_balancer.update_kl_coeff(self.global_step, args.kl_anneal_portion * args.num_total_iter,
                                         args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)
        kl_coeff = self.kl_balancer.kl_coeff

        # probability of drawing sample x from dist output by decoder
        loss_recon0 = recon_loss_fn(output_dist, target0, zero_alpha=0.1)
        loss_recon2 = recon_loss_fn(
            output_dist2, target2, zero_alpha=0.1, reduce=False)
        loss_recon2 = loss_recon2.sum(dim=(1, 2, 3, 4))

        balanced_kl, kl_coeffs, kl_vals = self.kl_balancer.balance(kl_all)

        w_res0 = 1.  # weight for res0 loss by value (8x voxels compared to res2)

        steric_output = output_dist
        loss_recon = loss_recon2 + loss_recon0 * w_res0

        loss_steric = torch.zeros(len(data['target_density_prot0']))

        if self.args.steric_loss_w > 0:
            loss_steric = steric_loss_fn(
                steric_output, data['target_density_prot0'], mean_only=self.args.steric_mean_only)

        nelbo_batch = loss_recon + balanced_kl
        loss = torch.mean(nelbo_batch)
        loss_spec_norm = spec_norm_loss_fn(model)
        loss_bn = batchnorm_loss_fn(model)

        # get spectral regularization coefficient (lambda)
        if args.weight_decay_norm_anneal:
            assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
            decay_norm_w = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + \
                kl_coeff * np.log(args.weight_decay_norm)
            decay_norm_w = np.exp(decay_norm_w)
        else:
            decay_norm_w = args.weight_decay_norm

        loss += (loss_spec_norm + loss_bn) * decay_norm_w + \
            torch.mean(loss_steric) * self.args.steric_loss_w
        self.nelbo_vals += loss.data
        self.nelbo_count += 1
        self.nelbo_avg = self.nelbo_vals/self.nelbo_count

        if (self.global_step + 1) % 10 == 0:
            # Logging to TensorBoard by default
            self.log("train_loss", loss)

            # mae
            mae0 = nn.functional.l1_loss(
                output_dist.sample_vals_mean(), target0, reduction='none')
            mae2 = nn.functional.l1_loss(
                output_dist2.sample_vals_mean(), target2, reduction='none')
            self.log('mae/mae0', mae0.mean(),
                     on_step=True, rank_zero_only=True)
            self.log('mae/mae2', mae2.mean(),
                     on_step=True, rank_zero_only=True)
            self.log('mae/mae0_nonzero',
                     mae0[target0 > 0.01].mean(), on_step=True, rank_zero_only=True)
            self.log('mae/mae2_nonzero',
                     mae2[target2 > 0.01].mean(), on_step=True, rank_zero_only=True)

            if mae2.shape[1] > 1:
                for i in range(mae2.shape[1]):
                    self.log(
                        f'mae/mae2_{i}', mae2[:, i].mean(), on_step=True, rank_zero_only=True)
                    self.log(f'mae/mae2_nonzero_{i}', mae2[:, i][target2[:, i] > 0.01].mean(
                    ), on_step=True, rank_zero_only=True)
            if mae0.shape[1] > 1:
                for i in range(mae0.shape[1]):
                    self.log(
                        f'mae/mae0_{i}', mae0[:, i].mean(), on_step=True, rank_zero_only=True)
                    self.log(f'mae/mae0_nonzero_{i}', mae0[:, i][target2[:, i] > 0.01].mean(
                    ), on_step=True, rank_zero_only=True)

            self.log('train/noise_dec', noise_dec,
                     on_step=True, rank_zero_only=True)
            self.log('train/norm_loss', loss_spec_norm,
                     on_step=True, rank_zero_only=True)
            self.log('train/bn_loss', loss_bn,
                     on_step=True, rank_zero_only=True)
            self.log('train/norm_coeff', decay_norm_w,
                     on_step=True, rank_zero_only=True)
            self.log('train/steric_loss', torch.mean(loss_steric),
                     on_step=True, rank_zero_only=True)

            self.log('train/nelbo_avg', self.nelbo_avg,
                     on_step=True, sync_dist=True)
            self.log('train/lr', self.optimizers().state_dict()
                     ['param_groups'][0]['lr'], on_step=True, rank_zero_only=True, prog_bar=True)
            self.log('train/nelbo_iter', loss,
                     on_step=True, rank_zero_only=True)
            self.log('train/kl_iter', torch.mean(sum(kl_all)),
                     on_step=True, rank_zero_only=True, prog_bar=False)
            self.log('train/recon_iter', torch.mean(loss_recon),
                     on_step=True, rank_zero_only=True, prog_bar=False)
            self.log('train/recon_iter0', torch.mean(loss_recon0), on_step=True,
                     rank_zero_only=True, prog_bar=False)  # full resolution
            self.log('train/recon_iter2', torch.mean(loss_recon2), on_step=True,
                     rank_zero_only=True, prog_bar=False)  # half resolution
            self.log('kl_coeff/coeff', float(kl_coeff),
                     on_step=True, rank_zero_only=True)

            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                if args.distributed:
                    average_distributed(kl_diag_i)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                self.log('kl/active_%d' % i, float(num_active),
                         on_step=True, rank_zero_only=True)
                self.log('kl_coeff/layer_%d' %
                         i, float(kl_coeffs[i]), on_step=True, rank_zero_only=True)
                self.log('kl_vals/layer_%d' %
                         i, float(kl_vals[i]), on_step=True, rank_zero_only=True)
                self.log('kl/total_active', float(total_active),
                         on_step=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        model = self.hvae_model
        args = self.args

        if batch_idx == 0:
            torch.cuda.set_device(self.device)
            self.kl_balancer.to_device(self.device)
        data = batch

        if 'input_density_lig0' not in data.keys():
            data['input_density_lig'] = self.gauss_conv_layer_lig(
                data['input_encoding_lig'])
            data['target_density_lig'] = data['input_density_lig']

            data['input_density_prot'] = self.gauss_conv_layer_prot(
                data['input_encoding_prot'])
            data['target_density_prot'] = data['input_density_prot']

            if args.input_grid_format2 == 'density' or args.output_grid_format2 == 'density':
                data['input_density_lig2'] = torch.nn.functional.interpolate(
                    data['input_density_lig'], scale_factor=0.5, mode='trilinear')
                data['target_density_lig2'] = data['input_density_lig2']
                if args.input_grid_format2 == 'density':
                    data['input_density_prot2'] = torch.nn.functional.interpolate(
                        data['input_density_prot'], scale_factor=0.5, mode='trilinear')
                    data['target_density_prot2'] = data['input_density_prot2']

            # cumulative density/encoding
            element_channels_lig = self.dataset.grid_encoder_ligand.channels_elements_only
            element_channels_prot = self.dataset.grid_encoder_protein.channels_elements_only
            data['input_density_lig'] = data['input_density_lig0'] = data['input_density_lig'][:,
                                                                                               element_channels_lig].sum(dim=1, keepdim=True)
            data['target_density_lig'] = data['target_density_lig0'] = data['input_density_lig']
            data['input_density_prot'] = data['input_density_prot0'] = data['input_density_prot'][:,
                                                                                                  element_channels_prot].sum(dim=1, keepdim=True)
            data['target_density_prot'] = data['target_density_prot0'] = data['input_density_prot']

            target0 = data['target_density_lig0']
            target2 = data['target_density_lig2']

            data['input_encoding_lig'] = data['target_encoding_lig'] = data['input_encoding_lig'].cpu()
            data['input_encoding_prot'] = data['target_encoding_prot'] = data['input_encoding_prot'].cpu()

        noise_dec = 0.
        if args.decoder_noise > 0:
            noise_dec = min(1., self.global_step / args.warmup_iters) * \
                args.decoder_noise  # warm-up decoder noise

        (out_distp0, out_distp2), log_q, log_p, kl_all, kl_diag = model(
            data, noise_dec=noise_dec)

        output_dist = model.dist_from_output(out_distp0)
        output_dist2 = model.dist_from_output(out_distp2)

        self.kl_balancer.update_kl_coeff(self.global_step, args.kl_anneal_portion * args.num_total_iter,
                                         args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)
        kl_coeff = self.kl_balancer.kl_coeff

        # probability of drawing sample x from dist output by decoder
        loss_recon0 = recon_loss_fn(output_dist, target0, zero_alpha=0.1)
        loss_recon2 = recon_loss_fn(
            output_dist2, target2, zero_alpha=0.1, reduce=False)
        loss_recon2 = loss_recon2.sum(dim=(1, 2, 3, 4))

        balanced_kl, kl_coeffs, kl_vals = self.kl_balancer.balance(kl_all)

        w_res0 = 1.  # weight for res0 loss

        steric_output = output_dist
        loss_recon = loss_recon2 + loss_recon0 * w_res0

        loss_steric = torch.zeros(len(data['target_density_prot0']))

        if self.args.steric_loss_w > 0:
            loss_steric = steric_loss_fn(
                steric_output, data['target_density_prot0'], mean_only=self.args.steric_mean_only)

        nelbo_batch = loss_recon + balanced_kl
        loss = torch.mean(nelbo_batch)
        loss_spec_norm = spec_norm_loss_fn(model)
        loss_bn = batchnorm_loss_fn(model)

        # get spectral regularization coefficient (lambda)
        if args.weight_decay_norm_anneal:
            assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
            decay_norm_w = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + \
                kl_coeff * np.log(args.weight_decay_norm)
            decay_norm_w = np.exp(decay_norm_w)
        else:
            decay_norm_w = args.weight_decay_norm

        loss += (loss_spec_norm + loss_bn) * decay_norm_w + \
            torch.mean(loss_steric) * self.args.steric_loss_w
        self.nelbo_vals_valid += loss.data
        self.nelbo_count_valid += 1
        self.nelbo_avg_valid = self.nelbo_vals_valid/self.nelbo_count_valid

        if batch_idx % 10 == 0:
            # Logging to TensorBoard by default
            on_epoch = True
            on_step = False
            rank_zero_only = False
            sync_dist = True

            batch_size = len(target0)
            self.log("valid/loss", loss, on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=True)


            self.log('valid/noise_dec', noise_dec, on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)
            self.log('valid/norm_loss', loss_spec_norm, on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)
            self.log('valid/bn_loss', loss_bn, on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)
            self.log('valid/norm_coeff', decay_norm_w, on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)
            self.log('valid/steric_loss', torch.mean(loss_steric), on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)

            self.log('valid/nelbo_avg', self.nelbo_avg_valid, on_step=on_step,
                     on_epoch=on_epoch, batch_size=batch_size, sync_dist=True)
            self.log('valid/lr', self.optimizers().state_dict()['param_groups'][0]['lr'], on_step=on_step,
                     on_epoch=on_epoch, batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist, prog_bar=True)
            self.log('valid/nelbo_iter', loss, on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)
            self.log('valid/kl_iter', torch.mean(sum(kl_all)), on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist, prog_bar=False)
            self.log('valid/recon_iter', torch.mean(loss_recon), on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist, prog_bar=False)
            self.log('valid/recon_iter0', torch.mean(loss_recon0), on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist, prog_bar=False)  # full resolution
            self.log('valid/recon_iter2', torch.mean(loss_recon2), on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist, prog_bar=False)  # half resolution
            self.log('kl_coeff/coeff', float(kl_coeff), on_step=on_step, on_epoch=on_epoch,
                     batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)

            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                if args.distributed:
                    average_distributed(kl_diag_i)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                self.log('kl/active_%d' % i, float(num_active), on_step=on_step, on_epoch=on_epoch,
                         batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)
                self.log('kl_coeff/layer_%d' % i, float(kl_coeffs[i]), on_step=on_step, on_epoch=on_epoch,
                         batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)
                self.log('kl_vals/layer_%d' % i, float(kl_vals[i]), on_step=on_step, on_epoch=on_epoch,
                         batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)
                self.log('kl/total_active', float(total_active), on_step=on_step, on_epoch=on_epoch,
                         batch_size=batch_size, rank_zero_only=rank_zero_only, sync_dist=sync_dist)

        # make scatterplots of pearson correlation
        scat_vals0 = [0, output_dist.sample_vals().detach().cpu(),
                      target0.detach().cpu(), '']
        scat_vals2 = [2, output_dist2.sample_vals().detach().cpu(),
                      target2.detach().cpu(), '']
        self.validation_step_outputs[0].append(scat_vals0)
        self.validation_step_outputs[1].append(scat_vals2)

        return loss

    def on_validation_epoch_end(self):
        scat_vals0, scat_vals2 = self.validation_step_outputs

        for scat_vals in [scat_vals0, scat_vals2]:
            out_i = scat_vals[0][0]
            pred_grid = torch.cat([x[1] for x in scat_vals])
            true_grid = torch.cat([x[2] for x in scat_vals])

            if dist.get_world_size() > 1:
                pred_grids_all = [torch.zeros_like(
                    pred_grid) for _ in range(dist.get_world_size())]
                true_grid_all = [torch.zeros_like(
                    true_grid) for _ in range(dist.get_world_size())]
                dist.all_gather(pred_grids_all, pred_grid)
                dist.all_gather(true_grid_all, true_grid)
                pred_grid = torch.cat(pred_grids_all)
                true_grid = torch.cat(true_grid_all)

            gridlabel = scat_vals[0][3]

            n_prs = 0

            label = str(out_i) + gridlabel
            true_vals, pred_vals = filter_predictions(
                true_grid.detach().cpu(), pred_grid.detach().cpu())

            pr_all = 0
            for ch in range(len(true_vals)):
                try:
                    pr, pvals = pearsonr(true_vals[ch], pred_vals[ch])
                except Exception as e:
                    pr = float('nan')
                    print(
                        f'Warning: Calculating PearsonR failed! label: {label}')

                if pr == pr:
                    n_prs += 1
                    pr_all += pr
                    self.log('valid_corr/pearson_r%s_%d' % (label, ch),
                             pr, prog_bar=False, rank_zero_only=True)
            if n_prs > 0:
                self.log('valid/pearson_r%s' % label, pr_all/n_prs,
                         prog_bar=('encoding' not in label), rank_zero_only=True)

            savedir = os.path.join(self.logger.log_dir, 'img', 'scatter')
            os.makedirs(savedir, exist_ok=True)

            if out_i == 0:
                channels = ['cumulative']
                colors = ['grey']
                alpha = 0.3
            else:
                channels = self.dataset.grid_encoder_ligand.channels
                colors = self.dataset.grid_encoder_ligand.channel_colors
                alpha = 0.7

            fig, axs = plot_scatterplot(
                true_vals, pred_vals, channels=channels, colors=colors, alpha=alpha)
            fig.savefig(os.path.join(savedir, 'scat%s_e%d_s%d.png' %
                        (label, self.current_epoch, self.global_step)))
            plt.close(fig)

        self.validation_step_outputs.clear()
        self.validation_step_outputs = [[], []]

    def on_train_epoch_end(self):
        args = self.args

        if args.get('zinc_alpha_anneal', False):
            self.dataset.zinc_alpha = np.clip(
                self.dataset.zinc_alpha + args.zinc_alpha_anneal_rate, min=args.zinc_alpha_min, max=args.zinc_alpha_max)

        self.log('gradients/skipped_updates', self.skipped_updates,
                 on_epoch=True, rank_zero_only=True, sync_dist=True)
        return super().on_train_epoch_end()

    def on_train_start(self):
        torch.set_float32_matmul_precision('highest')

        args = self.args
        if args.continue_from_checkpoint is not None:
            # self.optimizers().load_state_dict(self.optimizers()._optimizer.state_dict())
            opt = self.optimizers()
            opt.param_groups = opt._optimizer.param_groups

        print('Optimizer learning rate:', '%.2e' %
              self.optimizers().param_groups[0]['lr'])

        if not hasattr(self, 'skipped_updates'):
            self.skipped_updates = 0

        self.optimizers().zero_grad()

    def on_before_backward(self, loss):
        return

    def on_after_backward(self) -> None:
        # If using mixed precision, the gradients have already been scaled here
        return super().on_after_backward()

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        # If using mixed precision, the gradients are already unscaled here
        grad_norm_total = get_gradients_norm(self.hvae_model)
        self.log('gradients/norm', grad_norm_total, on_step=True,
                 rank_zero_only=True, prog_bar=True)

        if (self.args.gradient_skip_val != -1 and grad_norm_total >= self.args.gradient_skip_val) or np.isnan(grad_norm_total) or np.isinf(grad_norm_total):
            self.skipped_updates += 1
            optimizer.zero_grad()

    def configure_optimizers(self):
        args = self.args
        self.opts = []
        self.schs = []

        hvae_optimizer = Adamax(self.hvae_model.parameters(), args.learning_rate,
                                weight_decay=args.weight_decay, eps=1e-3)

        # warm-up lr (slowly increase lr to initial_lr)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            hvae_optimizer, start_factor=1/args.warmup_iters, end_factor=1.0, total_iters=args.warmup_iters)

        # pause cosine annealing during warmup
        hvae_sch1 = torch.optim.lr_scheduler.ConstantLR(
            hvae_optimizer, factor=1, total_iters=1)
        hvae_sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            hvae_optimizer,
            float(args.epochs - args.warmup_epochs - 1),
            eta_min=args.learning_rate_min
        )
        hvae_scheduler = torch.optim.lr_scheduler.SequentialLR(
            hvae_optimizer, schedulers=[hvae_sch1, hvae_sch2], milestones=[args.warmup_epochs])

        self.opts.append(hvae_optimizer)

        self.schs.append({'scheduler': warmup_scheduler, 'interval': 'step'})
        self.schs.append({'scheduler': hvae_scheduler, 'interval': 'epoch'})

        return self.opts, self.schs

    def configure_datasets(self):
        args = self.args
        data_args = args.data_args

        # Load Dataset
        print('\nLoading Datasets....\n', flush=True)
        DataClass = eval(data_args.data_class)
        data_dict = data_args.dict

        data_dict.update(
            {k: v for k, v in args.dict.items() if '_grid_format' in k})
        self.dataset = DataClass(params=data_dict, multi_read=True)

        train_length = int(data_args.train_split * len(self.dataset))
        test_length = int(len(self.dataset) - train_length)
        self.trainset, self.testset = torch.utils.data.random_split(
            self.dataset, lengths=[train_length, test_length])

    def configure_datasets(self):
        args = self.args
        data_args = args.data_args

        # Load Dataset
        print('\nLoading Datasets....\n', flush=True)
        DataClass = eval(data_args.data_class)
        data_dict = data_args.dict
        data_dict.update(
            {k: v for k, v in args.dict.items() if '_grid_format' in k})
        self.dataset = DataClass(
            params=data_dict, multi_read=True, split='train')
        self.dataset_train = self.dataset

        params_valid = data_args.dict
        params_valid['zinc_alpha'] = 0  # don't use ZINC dataset for validation
        self.dataset_valid = DataClass(
            params=params_valid, multi_read=True, split='valid')

    def train_dataloader(self):
        batch_size = self.args.batch_size
        return torch.utils.data.DataLoader(self.dataset_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=self.args.trainer_args.num_workers,
                                           pin_memory=False,
                                           drop_last=True,
                                           persistent_workers=True,
                                           )

    def val_dataloader(self):
        batch_size = self.args.batch_size
        return torch.utils.data.DataLoader(self.dataset_valid,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=self.args.trainer_args.num_workers,
                                           pin_memory=False,
                                           drop_last=True,
                                           persistent_workers=True,
                                           )

    @torch.no_grad()
    def get_no_receptor_input(self):
        '''Gets input for case where there is not receptor.'''
        ddict, out_distp_dict, output, idx = self.get_example()
        dat = {'input_density_prot0': ddict['input_density_prot0'],
               'input_density_prot2': ddict['input_density_prot2']}
        for key in dat:
            dat[key][:] = 0
        return dat

    @torch.no_grad()
    def get_example(self, dataset='all', data_key=None, idx=None, verbose=False):
        '''Gets example from dataset.'''
        if data_key is None:
            data_key = self.args.dict.get('input_grid_format', 'density')

        if dataset == 'train':
            data = self.trainset
        elif dataset == 'test':
            data = self.testset
        else:
            data = self.dataset

        if idx is None:
            idx = np.random.randint(len(data))
        ddict = data[idx]

        if dataset == 'train' or dataset == 'test':
            idx = ddict.indices[idx]

        with autocast():
            if 'input_density_lig0' not in ddict.keys():
                for key in ddict.keys():
                    if any([k in key for k in ['_encoding', '_density']]):
                        if ddict[key].ndim == 4:
                            ddict[key] = ddict[key].unsqueeze(0)
                ddict['input_encoding_lig'] = ddict['input_encoding_lig'].cuda()
                ddict['input_density_lig'] = self.gauss_conv_layer_lig(
                    ddict['input_encoding_lig'])
                ddict['target_density_lig'] = ddict['input_density_lig']

                ddict['input_encoding_prot'] = ddict['input_encoding_prot'].cuda()
                ddict['input_density_prot'] = self.gauss_conv_layer_prot(
                    ddict['input_encoding_prot'])
                ddict['target_density_prot'] = ddict['input_density_prot']

                ddict['input_density_lig2'] = torch.nn.functional.interpolate(
                    ddict['input_density_lig'], scale_factor=0.5, mode='trilinear')
                ddict['target_density_lig2'] = ddict['input_density_lig2']
                ddict['input_density_prot2'] = torch.nn.functional.interpolate(
                    ddict['input_density_prot'], scale_factor=0.5, mode='trilinear')
                ddict['target_density_prot2'] = ddict['input_density_prot2']

                # cumulative density
                ddict['coords_lig'] = self.dataset.grid_encoder_ligand.channels_elements_only
                element_channels_lig = self.dataset.grid_encoder_ligand.channels_elements_only
                element_channels_prot = self.dataset.grid_encoder_protein.channels_elements_only
                ddict['input_density_lig'] = ddict['input_density_lig0'] = ddict['input_density_lig'][:,
                                                                                                      element_channels_lig].sum(dim=1, keepdim=True)
                ddict['target_density_lig'] = ddict['target_density_lig0'] = ddict['input_density_lig']
                ddict['input_density_prot'] = ddict['input_density_prot0'] = ddict['input_density_prot'][:,
                                                                                                         element_channels_prot].sum(dim=1, keepdim=True)
                ddict['target_density_prot'] = ddict['target_density_prot0'] = ddict['input_density_prot']

                if self.args.input_grid_format == 'encoding':
                    ddict['input_encoding_lig0'] = ddict['target_encoding_lig0'] = ddict['input_encoding_lig'][:,
                                                                                                               element_channels_lig].sum(dim=1, keepdim=True)
                    ddict['input_encoding_prot0'] = ddict['target_encoding_prot0'] = ddict['input_encoding_prot'][:,
                                                                                                                  element_channels_prot].sum(dim=1, keepdim=True)

                ddict['input_encoding_lig'] = ddict['target_encoding_lig'] = ddict['input_encoding_lig'].cpu()
                ddict['input_encoding_prot'] = ddict['target_encoding_prot'] = ddict['input_encoding_prot'].cpu()

        with autocast():
            out_distp, log_q, log_p, kl_all, kl_diag = self.hvae_model(ddict)

            out_distp_dict = {'out_distp2': out_distp[1]}
            if self.args.output_grid_format == 'encoding':
                C = out_distp[0].shape[1]//2
                out_distp[0][:, C:] = torch.clip(out_distp[0][:, C:], max=0.02)
                out_distp_dict['out_distp_encoding'] = out_distp[0]

                out_distp0_density = torch.cat([2*self.gauss_conv_layer_lig0(out_distp[0][:, :C] / 2 + 0.5) - 1,
                                                out_distp[0][:, C:]], dim=1)
                output0_density = self.hvae_model.dist_from_output(
                    out_distp0_density)
                out_distp_dict['out_distp_density'] = out_distp0_density

            else:
                out_distp_dict['out_distp_density'] = out_distp[0]

            output = {}
            for key in out_distp_dict:
                output[key.replace('out_distp', 'output')] = self.hvae_model.dist_from_output(
                    out_distp_dict[key])

        return ddict, out_distp_dict, output, idx


    def get_z_res_dict(self):
        '''Gets the size of each latent tensor as a dictionary organized by latent resolution.'''
        if not hasattr(self, 'z_res_dict'):
            with torch.no_grad() and autocast():
                ddict, out_distp, output, idx = self.get_example(idx=None)
                out_distp, zlist, plist = self.hvae_model.get_sample_latents(
                    ddict, t=1)
            layers = np.array([x.shape[-1] for x in zlist])
            z_res_dict = {'res_%d' % x: np.arange(
                len(layers))[layers == x] for x in np.unique(layers)}
            self.z_res_dict = z_res_dict
        else:
            z_res_dict = self.z_res_dict
        return z_res_dict


    def get_example_from_dir(self, prot_dir, lig_pattern='*ligand.sdf', prot_pattern='*protein.pdb', random_rot=False, random_trans=False, return_latents=False, transform=None):
        '''Loads and processes an example from a directory.'''
        ligfile = glob.glob(os.path.join(prot_dir, lig_pattern))
        protfile = glob.glob(os.path.join(prot_dir, prot_pattern))
        assert len(
            protfile) == 1, f'Directory contained multiple `*protein.pdb` files:\n  %s' % ("\n  ".join(protfile))
        assert len(
            ligfile) == 1, f'Directory contained multiple `*ligand.sdf `files:\n  %s' % ("\n  ".join(ligfile))
        return self.get_example_from_file(ligfile[0], protfile[0], random_rot=random_rot, random_trans=random_trans, return_latents=return_latents, transform=transform)


    def get_example_from_file(self, ligfile, protfile=None, prot_idx=None, ref_ligpath=None, random_rot=False, random_trans=False, return_latents=False, transform=None):
        '''Loads and processes an example from ligand and protein files.'''
        assert protfile is not None or prot_idx is not None, 'Must input either protfile or prot_idx'
        molparser = MolParser()

        coords_lig, types_lig = molparser.mol_data_from_file(ligfile)
        coords_prot, types_prot = molparser.mol_data_from_file(protfile)

        if ref_ligpath is not None:
            coords_ref, _ = molparser.mol_data_from_file(ref_ligpath)
        else:
            coords_ref = coords_lig

        center_ref = coords_ref.mean(axis=0)
        coords_lig -= center_ref
        coords_prot -= center_ref

        # filter out protein atoms beyond a distance threshold
        dist_thresh = self.dataset.grid_encoder_ligand.grid_size
        # print('\nprot_dists', np.linalg.norm(coords_prot, axis=-1))
        prot_mask = np.linalg.norm(coords_prot, axis=-1) < dist_thresh

        coords_prot = coords_prot[prot_mask]
        types_prot = types_prot[prot_mask]
        return self.get_example_from_coords_types(coords_lig, types_lig, coords_prot, types_prot, random_rot=random_rot, random_trans=random_trans, return_latents=return_latents, transform=transform)

    def get_example_from_coords_types(self,
                                      atom_coords_lig,
                                      atom_types_lig,
                                      atom_coords_prot=None,
                                      atom_types_prot=None,
                                      prot_idx=None,
                                      protfile=None,
                                      ligfile=None,
                                      random_rot=False,
                                      random_trans=False,
                                      data_key=None,
                                      return_latents=False,
                                      transform=None,
                                      verbose=False):
        '''Processes an example from coordinates and types for ligand and protein.'''
        if data_key is None:
            data_key = self.args.dict.get('input_grid_format', 'density')

        assert (atom_coords_prot is not None and atom_types_prot is not None) or (prot_idx is not None) or (
            protfile is not None and ligfile is not None), 'Must provide prot_idx or protein coords and types.'

        data = self.dataset

        ge_lig = self.dataset.grid_encoder_ligand.copy()

        if protfile is not None:
            molparser = MolParser()
            coords_lig, _ = molparser.mol_data_from_file(ligfile)
            atom_coords_prot, atom_types_prot = molparser.mol_data_from_file(
                protfile)
            atom_coords_prot -= coords_lig.mean(axis=0)

        if transform is None:
            transform = {}
            if random_rot:
                atom_coords_lig, angles, success = rot3d_random_safe(
                    atom_coords_lig, ge_lig)
                atom_coords_prot = rot3d(atom_coords_prot, angles)
                transform['rot'] = angles
            if random_trans:
                atom_coords_lig, trans, success = trans3d_random_safe(
                    atom_coords_lig, ge_lig, max_dist=1, voxels=False)  # max dist in angstroms
                atom_coords_prot = trans3d(atom_coords_prot, trans)
                transform['trans'] = trans
        else:
            # print('transforming coordinates from input...')
            if 'rot' in transform:
                atom_coords_lig = rot3d(atom_coords_lig, transform['rot'])
                atom_coords_prot = rot3d(atom_coords_prot, transform['rot'])
            if 'trans' in transform:
                atom_coords_lig = trans3d(atom_coords_lig, transform['trans'])
                atom_coords_prot = trans3d(
                    atom_coords_prot, transform['trans'])
            if 'matrix' in transform:
                atom_coords_lig = atom_coords_lig @ transform['matrix']
                atom_coords_prot = atom_coords_prot @ transform['matrix']
                
        if prot_idx is not None:
            ddict = data[prot_idx]
        else:
            ddict = {}
            ge_prot = self.dataset.grid_encoder_protein.copy()
            ge_prot.atom_types = atom_types_prot
            ge_prot.atom_coords = atom_coords_prot
            ge_prot.encode_coords2grid()
            ddict['input_encoding_prot'] = ge_prot.values

        ge_lig.atom_types = atom_types_lig
        ge_lig.atom_coords = atom_coords_lig
        ge_lig.encode_coords2grid()
        ddict['input_encoding_lig'] = ge_lig.values

        ddict['coords_lig'] = atom_coords_lig
        ddict['coords_prot'] = atom_coords_prot

        with autocast():
            if 'input_density_lig0' not in ddict.keys():
                for key in ddict.keys():
                    if any([k in key for k in ['_encoding', '_density']]):
                        if ddict[key].ndim == 4:
                            ddict[key] = ddict[key].unsqueeze(0)
                ddict['input_encoding_lig'] = ddict['input_encoding_lig'].cuda()
                ddict['input_density_lig'] = self.gauss_conv_layer_lig(
                    ddict['input_encoding_lig'])
                ddict['target_density_lig'] = ddict['input_density_lig']

                ddict['input_encoding_prot'] = ddict['input_encoding_prot'].cuda()
                ddict['input_density_prot'] = self.gauss_conv_layer_prot(
                    ddict['input_encoding_prot'])
                ddict['target_density_prot'] = ddict['input_density_prot']

                ddict['input_density_lig2'] = torch.nn.functional.interpolate(
                    ddict['input_density_lig'], scale_factor=0.5, mode='trilinear')
                ddict['target_density_lig2'] = ddict['input_density_lig2']
                ddict['input_density_prot2'] = torch.nn.functional.interpolate(
                    ddict['input_density_prot'], scale_factor=0.5, mode='trilinear')
                ddict['target_density_prot2'] = ddict['input_density_prot2']

                # cumulative density
                element_channels_lig = self.dataset.grid_encoder_ligand.channels_elements_only
                element_channels_prot = self.dataset.grid_encoder_protein.channels_elements_only
                ddict['input_density_lig'] = ddict['input_density_lig0'] = ddict['input_density_lig'][:,
                                                                                                      element_channels_lig].sum(dim=1, keepdim=True)
                ddict['target_density_lig'] = ddict['target_density_lig0'] = ddict['input_density_lig']
                ddict['input_density_prot'] = ddict['input_density_prot0'] = ddict['input_density_prot'][:,
                                                                                                         element_channels_prot].sum(dim=1, keepdim=True)
                ddict['target_density_prot'] = ddict['target_density_prot0'] = ddict['input_density_prot']

                if self.args.input_grid_format == 'encoding':
                    ddict['input_encoding_lig0'] = ddict['target_encoding_lig0'] = ddict['input_encoding_lig'][:,
                                                                                                               element_channels_lig].sum(dim=1, keepdim=True)
                    ddict['input_encoding_prot0'] = ddict['target_encoding_prot0'] = ddict['input_encoding_prot'][:,
                                                                                                                  element_channels_prot].sum(dim=1, keepdim=True)

                ddict['input_encoding_lig'] = ddict['target_encoding_lig'] = ddict['input_encoding_lig'].cpu()
                ddict['input_encoding_prot'] = ddict['target_encoding_prot'] = ddict['input_encoding_prot'].cpu()

        with autocast():
            out_distp, zlist, plist = self.hvae_model.get_sample_latents(ddict)
            latents = {'z': zlist, 'p': plist}

            out_distp_dict = {'out_distp2': out_distp[1]}
            if self.args.output_grid_format == 'encoding':
                C = out_distp[0].shape[1]//2
                out_distp[0][:, C:] = torch.clip(out_distp[0][:, C:], max=0.02)
                out_distp_dict['out_distp_encoding'] = out_distp[0]

                out_distp0_density = torch.cat([2*self.gauss_conv_layer_lig0(out_distp[0][:, :C] / 2 + 0.5) - 1,
                                                out_distp[0][:, C:]], dim=1)
                output0_density = self.hvae_model.dist_from_output(
                    out_distp0_density)
                out_distp_dict['out_distp_density'] = out_distp0_density

            else:
                out_distp_dict['out_distp_density'] = out_distp[0]

            output = {}
            for key in out_distp_dict:
                output[key.replace('out_distp', 'output')] = self.hvae_model.dist_from_output(
                    out_distp_dict[key])

        if return_latents:
            return ddict, out_distp_dict, output, prot_idx, transform, latents
        else:
            return ddict, out_distp_dict, output, prot_idx, transform

    def init_lig_grids(self):
        '''Initializes grid encoders.'''
        lig_grid = self.dataset.grid_encoder_ligand.copy()
        lig_grid.channels = lig_grid.channels[:1]
        lig_grid.init_grid()
        lig_grid2 = self.dataset.grid_encoder_ligand.copy()
        return lig_grid, lig_grid2
