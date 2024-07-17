import numpy as np
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import NormalDist
from .blocks import ConvLayer, EncoderMixBlockProt, DecoderMixBlockProt, EncoderMixBlock, DecoderMixBlock, \
                               EncoderBlock, DecoderBlock, SamplerBlock, EncoderModule, \
                               DecoderModule, InputBlock, OutputBlock, Downsample, Upsample
                                

class AutoEncoderComplexSplit(nn.Module):
    def __init__(self, args):
        super(AutoEncoderComplexSplit, self).__init__()

        self.args = args

        self.args.decoder_noise = self.args.get('decoder_noise', 0.)

        args.n_channels_input = args.get('n_channels_input', 1)
        args.n_channels_input_prot = args.get('n_channels_input_prot', 1)
        args.n_channels_input2 = args.get('n_channels_input2', len(args.data_args.channels_in_lig))
        args.n_channels_input_prot2 = args.get('n_channels_input_prot2', len(args.data_args.channels_in_prot))

        self.ch_growth = args.ch_growth
        self.n_channels_input0 = args.n_channels_input
        self.n_channels_input_prot0 = args.n_channels_input_prot
        
        self.n_channels_start = args.n_channels_start
        self.n_channels_start_prot = args.n_channels_start_prot

        self.n_channels_end = args.n_channels_end
        self.n_channels_end_prot = args.n_channels_end_prot

        self.n_channels_input2 = args.n_channels_input2
        self.n_channels_input_prot2 = args.n_channels_input_prot2

        self.n_channels_start2 = args.n_channels_start2
        self.n_channels_start_prot2 = args.n_channels_start_prot2

        self.n_channels_end2 = args.n_channels_start2
        self.n_channels_end_prot2 = args.n_channels_start_prot2
        
        self.n_channels_input = args.n_channels_input + args.n_channels_input2  # concatenated input channels
        self.n_channels_input_prot = args.n_channels_input_prot + args.n_channels_input_prot2 # concatenated input channels


        self.combine_prot_enc = args.combine_prot_enc  # OPTIONS: block, group, module, input_only
        self.combine_prot_dec = args.combine_prot_dec  # OPTIONS: block, group, module
        assert (self.combine_prot_dec == self.combine_prot_enc) or self.combine_prot_enc, 'combine_prot_enc must be either be `input_only` or same as combine_prot_dec'
        
        self.n_encoder_stem_modules = args.n_encoder_stem_modules
        self.n_encoder_stem_modules2 = args.n_encoder_stem_modules2
        self.n_encoder_stem_blocks = args.n_encoder_stem_blocks  # convs per stem module

        self.n_decoder_stem_modules = args.n_decoder_stem_modules
        self.n_decoder_stem_blocks = args.n_decoder_stem_blocks  # convs per stem module
        self.n_decoder_stem_modules2 = args.get('n_decoder_stem_modules2', self.n_encoder_stem_modules2)

        self.n_convs_dec_block = args.n_convs_dec_block
        self.n_convs_enc_block = args.n_convs_enc_block

        self.n_encoder_modules = args.n_latent_res
        self.n_decoder_modules = args.n_latent_res
        self.n_encoder_group_blocks = args.n_encoder_group_blocks
        self.n_decoder_group_blocks = args.n_decoder_group_blocks

        self.n_latent_res = args.n_latent_res
        self.groups_per_res = args.groups_per_res
        self.n_group_latents = args.n_group_latents

        self.grid_size = args.grid_size

        n_downsample_tot = (self.n_encoder_stem_modules + self.n_latent_res - 1)
        M = self.ch_growth ** n_downsample_tot
        grid_res0_shape = [self.grid_size // 2 ** n_downsample_tot] * 3
        self.z0_shape = [self.n_group_latents] + grid_res0_shape   # for sampling
        self.prior_dec_input0 = nn.Parameter(torch.rand(size=[int(M * self.n_channels_end + (M/2) * self.n_channels_end2)] + grid_res0_shape), requires_grad=True)  # for first prior dist

        ################## MAIN ENCODER #####################

        # input module
        self.input_module = InputBlock(self.n_channels_input0 + self.n_channels_input_prot0, self.n_channels_start)
        self.input_module2 = InputBlock(self.n_channels_input2 + self.n_channels_input_prot2, self.n_channels_start2) # split resolution


        # encoder stem
        self.encoder_stem = nn.ModuleList()
        self.encoder_stem.channels_in = self.input_module.channels_out
        c_in = self.input_module.channels_out
        n_ch_prot0 = 2*self.n_channels_start + self.n_channels_start_prot2
        for i in range(self.n_encoder_stem_modules):
            self.encoder_stem.append(EncoderModule(c_in, c_in, self.n_encoder_stem_blocks, downsample=False, n_block_convs=self.n_convs_enc_block))
            if self.combine_prot_enc == 'group' or self.combine_prot_enc == 'scale':
                c_in_prot = n_ch_prot0 * self.ch_growth ** (i+1)
                self.encoder_stem.append(EncoderMixBlockProt(c_in, c_in_prot))
            
            # downsample
            self.encoder_stem.append(EncoderBlock(c_in, int(c_in * self.ch_growth), downsample=True, n_convs=self.n_convs_enc_block))
            c_in = int(c_in * self.ch_growth)
        self.encoder_stem.channels_out = self.encoder_stem[-1].channels_out


        # encoder main
        self.encoder_main = nn.ModuleList()
        self.encoder_main.channels_in = self.encoder_stem.channels_out + self.n_channels_start2
        c_in = self.encoder_main.channels_in
        n_ch_dec0 = 2*self.n_channels_end + self.n_channels_end2
        for i in range(self.n_latent_res):
            last_s = (i == self.n_latent_res - 1)
            M = self.ch_growth ** (self.n_encoder_stem_modules + i) # channel multiplier
            c_in_prot = int(n_ch_prot0 * M)
            for j in range(self.groups_per_res[i]):
                last_g = (j == self.groups_per_res[i] - 1)
                self.encoder_main.append(EncoderModule(c_in, c_in, self.n_decoder_group_blocks, downsample=False, n_block_convs=self.n_convs_enc_block))
                if not (last_g and last_s):
                    self.encoder_main.append(EncoderMixBlock(c_in, c_in))
                if self.combine_prot_enc == 'group':
                    self.encoder_main.append(EncoderMixBlockProt(c_in, c_in_prot))
            
            if self.combine_prot_enc == 'scale' and not last_s:
                self.encoder_main.append(EncoderMixBlockProt(c_in, c_in_prot))

            if not last_s:   # downsample
                self.encoder_main.append(EncoderBlock(c_in, int(c_in * self.ch_growth), downsample=True, n_convs=self.n_convs_enc_block))
                c_in = c_in * self.ch_growth
        self.encoder_main.channels_out = self.encoder_main[-1].channels_out

        
        ################## PROTEIN ENCODER #####################
        
        # input prot mix
        if not (self.combine_prot_enc == 'input_only'):
            self.input_mix_prot = EncoderMixBlockProt(self.n_channels_start, self.n_channels_start_prot)
        else:
            self.input_mix_prot = nn.Identity()


        # input module prot
        self.input_module_prot = InputBlock(self.n_channels_input_prot0, self.n_channels_start_prot)
        self.input_module_prot2 = InputBlock(self.n_channels_input_prot2, self.n_channels_start_prot2) # split resolution


        # encoder stem prot
        self.encoder_stem_prot = nn.ModuleList()
        self.encoder_stem_prot.channels_in = self.input_module_prot.channels_out
        c_in = self.input_module_prot.channels_out
        for i in range(self.n_encoder_stem_modules):
            self.encoder_stem_prot.append(EncoderModule(c_in, c_in, self.n_encoder_stem_blocks, downsample=False, n_block_convs=self.n_convs_enc_block))
            if self.combine_prot_enc == 'group' or self.combine_prot_enc == 'scale':
                self.encoder_stem_prot.append(nn.Identity())  # placeholder to match encoder stem
            
            # downsample
            self.encoder_stem_prot.append(EncoderBlock(c_in, int(c_in * self.ch_growth), downsample=True, n_convs=self.n_convs_enc_block))
            c_in = int(c_in * self.ch_growth)
        self.encoder_stem_prot.channels_out = self.encoder_stem_prot[-1].channels_out


        # encoder main prot
        self.encoder_main_prot = nn.ModuleList()
        self.encoder_main_prot.channels_in = self.encoder_stem_prot.channels_out + self.n_channels_start_prot2
        c_in = self.encoder_main_prot.channels_in
        for i in range(self.n_latent_res):
            last_s = (i == self.n_latent_res - 1)
            for j in range(self.groups_per_res[i]):
                last_g = (j == self.groups_per_res[i] - 1)
                self.encoder_main_prot.append(EncoderModule(c_in, c_in, self.n_encoder_group_blocks, downsample=False, n_block_convs=self.n_convs_enc_block))

            if not last_s:   # downsample
                self.encoder_main_prot.append(EncoderBlock(c_in, int(c_in * self.ch_growth), downsample=True, n_convs=self.n_convs_enc_block))
                c_in = c_in * self.ch_growth
        self.encoder_main_prot.channels_out = self.encoder_main_prot[-1].channels_out

        # encoder foot
        self.encoder_foot = nn.Identity()
        self.encoder_foot.channels_in = self.encoder_main.channels_out
        self.encoder_foot.channels_out = self.encoder_main.channels_out


        ################## MAIN DECODER #####################

        
        # decoder main
        self.decoder_main = nn.ModuleList()
        self.decoder_main.channels_in = self.encoder_foot.channels_out
        c_in = self.decoder_main.channels_in

        # start with latent and prot mixing
        M = self.ch_growth ** (self.n_latent_res + self.n_decoder_stem_modules - 2)  # channel multiplier
        c_in_prot = int(self.n_channels_end_prot * M * self.ch_growth + self.n_channels_end_prot2 * M)
        self.decoder_main.append(DecoderMixBlockProt(c_in, c_in_prot))

        for i in range(self.n_latent_res):
            last_s = (i == self.n_latent_res - 1)
            for j in range(self.groups_per_res[i]):
                last_g = (j == self.groups_per_res[i] - 1)
                self.decoder_main.append(DecoderModule(c_in, c_in, self.n_decoder_group_blocks, upsample=False, n_block_convs=self.n_convs_dec_block))

                self.decoder_main.append(DecoderMixBlock(c_in, self.n_group_latents))
                
                M = self.ch_growth ** (self.n_latent_res + self.n_decoder_stem_modules - i - 2)  # channel multiplier
                c_in_prot = int(self.n_channels_end_prot * M * self.ch_growth + self.n_channels_end_prot2 * M)
                if last_g:
                    c_in_prot = int(c_in_prot / 2)  # mix occurs before downsample (and channel growth)
                if (self.combine_prot_dec == 'group') and not last_g:
                    self.decoder_main.append(DecoderMixBlockProt(c_in, c_in_prot))
                    
            if not last_s:   # upsample
                self.decoder_main.append(DecoderBlock(c_in, c_in // self.ch_growth, upsample=True, n_convs=self.n_convs_dec_block))
                c_in = c_in // self.ch_growth
            
            # combine after upsample
            if last_s:
                c_in_prot = self.encoder_stem_prot.channels_out  # int(c_in_prot * self.ch_growth)
            self.decoder_main.append(DecoderMixBlockProt(c_in, c_in_prot))   
        self.decoder_main.channels_out = self.decoder_main[-1].channels_out


        # decoder stem       
        self.decoder_stem = nn.ModuleList()
        self.decoder_stem.channels_in = self.decoder_main.channels_out
        c_in = self.decoder_stem.channels_in
        c_out = self.encoder_stem.channels_in * self.ch_growth ** (self.n_encoder_stem_modules)
        self.decoder_stem.append(ConvLayer(c_in, c_out, kernel_size=1, padding=0, bias=True))
        self.decoder_stem[-1].channels_in = c_in
        self.decoder_stem[-1].channels_out = c_out
        c_in = c_out
        for i in range(self.n_encoder_stem_modules):
            c_out = c_in // self.ch_growth
            self.decoder_stem.append(DecoderModule(c_in, c_out, self.n_decoder_stem_blocks, upsample=True, n_block_convs=self.n_convs_dec_block))
            c_in = c_out
        self.decoder_stem.channels_out = c_out


        # decoder stem res2        
        self.decoder_stem2 = nn.ModuleList()
        self.decoder_stem2.channels_in = self.decoder_main.channels_out
        c_in = self.decoder_stem2.channels_in
        c_out = self.decoder_stem2.channels_in
        self.decoder_stem2.append(ConvLayer(c_in, c_out, kernel_size=1, padding=0, bias=True))
        self.decoder_stem2[-1].channels_in = c_in
        self.decoder_stem2[-1].channels_out = c_out
        c_out = c_in
        for i in range(self.n_decoder_stem_modules2):
            c_out = c_in if (i < self.n_decoder_stem_modules2 - 1) else self.n_channels_end2
            self.decoder_stem2.append(DecoderModule(c_in, c_out, self.n_decoder_stem_blocks, upsample=False, n_block_convs=self.n_convs_dec_block))
            c_in = c_out
        self.decoder_stem2.channels_out = c_out


        # output module
        self.output_module = OutputBlock(self.decoder_stem.channels_out, 2 * self.n_channels_input0)
        self.output_module2 = OutputBlock(self.decoder_stem2.channels_out, 2 * self.n_channels_input2) # split resolution


        # encoder latent sampler
        self.encoder_sampler = nn.ModuleList()
        c_in = self.encoder_foot.channels_out
        for i in range(self.n_latent_res):
            for j in range(self.groups_per_res[self.n_latent_res - i - 1]):
                self.encoder_sampler.append(SamplerBlock(c_in, 2 * self.n_group_latents, act=False))
            c_in = c_in // self.ch_growth


        # decoder latent sampler
        self.decoder_sampler = nn.ModuleList()
        c_in = self.encoder_foot.channels_out
        for i in range(self.n_latent_res):
            for j in range(self.groups_per_res[self.n_latent_res - i - 1]):
                if not (i == 0 and j == 0):  # fixed Normal for first group
                    self.decoder_sampler.append(SamplerBlock(c_in, 2 * self.n_group_latents, act=True))
            c_in = c_in // self.ch_growth

        
        self.layers_log_norm_all = []
        self.layers_conv_all = []
        self.layers_bn_all = []
        for _, layer in self.named_modules():
            if isinstance(layer, ConvLayer):
                self.layers_log_norm_all.append(layer.log_wn)
                self.layers_conv_all.append(layer)

            if isinstance(layer, nn.BatchNorm3d):
                self.layers_bn_all.append(layer)

        print('len log norm:', len(self.layers_log_norm_all))
        print('len bn:', len(self.layers_bn_all))

        self.spectral_norm_u = {}
        self.spectral_norm_v = {}


    def dist_from_output(self, dist_params):
        return NormalDist.init_from_dist_params(dist_params)


    def forward(self, xdict, noise_dec=0):
        return self._forward(xdict, t=1., noise_dec=noise_dec)
    

    def get_sample_latents(self, xdict, t=1., q_only=False):
        return self._forward(xdict, t, return_latents=True, q_only=q_only)
    

    def _forward(self, xdict, t=1., return_latents=False, q_only=False, noise_dec=0):
        if self.args.get('input_grid_format2', 'density') == 'encoding':
            x_lig2 = xdict['input_encoding_lig2']
            x_prot2 = xdict['input_encoding_prot2']
        elif self.args.get('input_grid_format2', 'density') == 'density':
            x_lig2 = xdict['input_density_lig2']
            x_prot2 = xdict['input_density_prot2']
        
        if self.args.input_grid_format == 'density':
            x_lig0 = xdict['input_density_lig0']
            x_prot0 = xdict['input_density_prot0']
        elif self.args.input_grid_format == 'encoding':
            x_lig0 = xdict['input_encoding_lig0']
            x_prot0 = xdict['input_encoding_prot0']
        
        zlist = []
        plist = []

        s_prot2 = 2 * x_prot2 - 1.0
        s_lig2 = 2 * x_lig2 - 1.0

        s_prot2_all = []
        s_lig2_all = []

        s_prot2_all.append(s_prot2)
        s_lig2_all.append(s_lig2)

        s_lig2 = self.input_module2(torch.cat([s_lig2, s_prot2], dim=1))
        s_prot2 = self.input_module_prot2(s_prot2)
        
        s_prot2_all.append(s_prot2)
        s_lig2_all.append(s_lig2)


        s_prot_all = []
        s_prot = 2 * x_prot0 - 1.0
        s_prot_all.append(s_prot)

        s = 2 * x_lig0 - 1.0
        s = self.input_module(torch.cat([s, s_prot], dim=1))

        s_prot = self.input_module_prot(s_prot)
        s_prot_all.append(s_prot)

        if not (self.combine_prot_enc == 'input_only'):
            s = self.input_mix_prot(s, s_prot)
            

        # run encoder stem prot
        encoder_stem_prot_s = []
        for mod in self.encoder_stem_prot:
            s_prot = mod(s_prot)
            if isinstance(mod, EncoderBlock) and mod.downsample:
                encoder_stem_prot_s.append(s_prot)

        s_prot_all.extend(encoder_stem_prot_s)


        # run encoder stem
        pi = 0
        for mod in self.encoder_stem:
            if isinstance(mod, EncoderMixBlockProt):
                s = mod(s, encoder_stem_prot_s[pi])
                pi += 1
            else:
                s = mod(s)


        # combine res2 branch and res1 branch
        s_prot = torch.cat([s_prot, s_prot2], dim=1)
        s = torch.cat([s, s_lig2], dim=1)


        # run encoder main prot
        encoder_main_prot_s = []
        for mi, mod in enumerate(self.encoder_main_prot):
            next_mod = self.encoder_main_prot[mi + 1] if mi < len(self.encoder_main_prot) - 1 else None
            s_prot = mod(s_prot)
            if self.combine_prot_dec == 'scale':
                if mi == len(self.encoder_main_prot) - 1 or getattr(mod, 'downsample', False):
                    encoder_main_prot_s.append(s_prot)
            elif self.combine_prot_dec == 'group' and isinstance(mod, EncoderModule): # not getattr(next_mod, 'downsample', False):
                encoder_main_prot_s.append(s_prot)
        # s_prot_all.extend(encoder_main_prot_s[:-1])
        s_prot_all.extend(encoder_main_prot_s)


        if self.combine_prot_enc == 'input_only':
            prot_s_for_enc_main = [encoder_main_prot_s[-1]]
        else:
            prot_s_for_enc_main = encoder_main_prot_s

        # run the main encoder tower
        mods_mix_enc = []
        mods_mix_s = []
        pi = 0
        for mod in self.encoder_main:
            if isinstance(mod, EncoderMixBlock):
                mods_mix_enc.append(mod)
                mods_mix_s.append(s)
            elif isinstance(mod, EncoderMixBlockProt):
                s = mod(s, prot_s_for_enc_main[pi])
                pi += 1
            else:
                s = mod(s)

        # reverse for running decoder
        mods_mix_enc.reverse()
        mods_mix_s.reverse()
        s_prot_all.reverse()
    

        di = 0
        dec_input0 = self.encoder_foot(s) 
        dist_params0 = self.encoder_sampler[di](dec_input0)
        mu_q, log_sigma_q = torch.chunk(dist_params0, 2, dim=1)
        dist = NormalDist(mu_q, log_sigma_q)
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)

        all_q = [dist]
        all_log_q = [log_q_conv]

        s = 0  # for safety

        # first prior distribution
        dist = NormalDist(mu=torch.zeros_like(z), log_sigma=torch.zeros_like(z))
        log_p_conv = dist.log_p(z)
        all_p = [dist]
        all_log_p = [log_p_conv]
        
        if return_latents:
            zlist.append(z.detach().clone())
            plist.append((mu_q.detach().clone(), log_sigma_q.detach().clone()))

        di = 0
        s = self.prior_dec_input0.unsqueeze(0)
        n_batch = z.size(0)
        s = s.expand(n_batch, -1, -1, -1, -1)

        pi = 0
        for mi, mod in enumerate(self.decoder_main):
            if isinstance(mod, DecoderMixBlockProt):
                try:
                    s = mod(s, s_prot_all[pi])
                except Exception as e:
                    print(mi, pi, s.shape, s_prot_all[pi].shape)
                    print('pi-1:', s_prot_all[pi-1].shape, '\tpi+1:', s_prot_all[pi+1].shape)
                    raise e

                pi += 1
            elif isinstance(mod, DecoderMixBlock):
                if di > 0:
                    # encoder prior dist params
                    enc_mix_out = mods_mix_enc[di - 1](mods_mix_s[di - 1], s)
                    dist_params = self.encoder_sampler[di](enc_mix_out)
                    mu_q, log_sigma_q = torch.chunk(dist_params, 2, dim=1)

                    # decoder prior dist params
                    dist_params = self.decoder_sampler[di - 1](s)
                    mu_p, log_sigma_p = torch.chunk(dist_params, 2, dim=1)
                    
                    dist = NormalDist(mu_p + mu_q, log_sigma_p + log_sigma_q)
                    z, _ = dist.sample()
                    log_q_conv = dist.log_p(z)

                    if return_latents:
                        zlist.append(z.detach().clone())
                        if q_only:
                            plist.append(((mu_q).detach().clone(), (log_sigma_q).detach().clone()))
                        else:
                            plist.append(((mu_q + mu_p).detach().clone(), (log_sigma_q + log_sigma_p).detach().clone()))

                    all_log_q.append(log_q_conv)
                    all_q.append(dist)

                    # get log prob of z
                    dist = NormalDist(mu_p, log_sigma_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)

                    # Add noise after each DecoderMixBlock to the mean
                    if noise_dec > 0:
                        z[:,:z.shape[1]//2] = z[:,:z.shape[1]//2] + (torch.randn_like(z[:,:z.shape[1]//2])) * noise_dec

                s = mod(s, z)
                di += 1
            else:
                s = mod(s)


        ### res2 output
        s2 = s
        for mod in self.decoder_stem2:
            s2 = mod(s2)
        out2 = self.output_module2(s2)
        ###


        # run decoder stem
        for mod in self.decoder_stem:
            if isinstance(mod, EncoderMixBlockProt):
                s = mod(s, s_prot_all[pi])
                pi += 1
            else:
                s = mod(s)

        outputs = (self.output_module(s), out2)

        if not return_latents:
            # kl divergence
            kl_all = []
            kl_diag = []
            log_p, log_q = 0., 0.
            for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
                kl_per_var = q.kl(p)
                kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3, 4]), dim=0))
                kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3, 4]))
                log_q += torch.sum(log_q_conv, dim=[1, 2, 3, 4])
                log_p += torch.sum(log_p_conv, dim=[1, 2, 3, 4])

        if return_latents:
            return outputs, zlist, plist
        else:
            return outputs, log_q, log_p, kl_all, kl_diag

    
    def sample(self, x_prot, num_samples=1, temps=1):
        '''Sample from prior.'''
        return self._sample(x_prot, num_samples, temps)
    
    def sample_near(self, x_prot, num_samples=1, temps=1, zbeta=0, plist=None, zlist=None):
        '''Sample near a given latent representation (prior-posterior interpolation).'''
        assert zlist or plist, 'Must give a list of `latents` (zlist) or list of `mu, log_sigma` (plist).'
        return self._sample(x_prot, num_samples, temps, zbeta, plist, zlist)
    
    def sample_spatial(self, x_prot, spatial_idxs, num_samples=1, temps=1, zbeta=0, plist=None, zlist=None, invert_spatial=False):
        '''Sample near a given latent representation over a spatial subset of latent space (spatial prior-posterior interpolation).'''
        return self._sample(x_prot, num_samples, temps, zbeta, plist, zlist, spatial_idxs=spatial_idxs, invert_spatial=invert_spatial)
    
    def _sample(self, x_prot, num_samples=1, temps=1, zbeta=0, plist=None, zlist=None, spatial_idxs=None, invert_spatial=False):
        assert not plist or not zlist, 'specify either plist or zlist, or neither. Not both.'

        n_latents = sum(self.groups_per_res)

        if not isinstance(temps, Iterable):
            temps = [temps]*n_latents
        if not isinstance(zbeta, Iterable):
            zbeta = [zbeta]*n_latents

        assert len(zbeta) == n_latents or len(zbeta) == self.n_latent_res, f'length of zbeta must be either {n_latents} or {self.n_latent_res}.'
                        
        di = 0
        up_level = 0

        z0_shape = [num_samples] + self.z0_shape

        if isinstance(x_prot, dict):
            x_prot0, x_prot2 = x_prot['input_density_prot0'], x_prot['input_density_prot2']
        else:
            x_prot0, x_prot2 = x_prot

        if num_samples > 1:
            x_prot0 = x_prot0.expand(num_samples, -1, -1, -1, -1)
            x_prot2 = x_prot2.expand(num_samples, -1, -1, -1, -1)

        s_prot2_all = []
        s_prot2 = 2 * x_prot2 - 1.0
        s_prot2_all.append(s_prot2)

        s_prot2 = self.input_module_prot2(s_prot2)
        s_prot2_all.append(s_prot2)

        ####
        s_prot_all = []
        s_prot = 2 * x_prot0 - 1.0
        s_prot_all.append(s_prot)

        s_prot = self.input_module_prot(s_prot)
        s_prot_all.append(s_prot)


        # run encoder stem prot
        encoder_stem_prot_s = []
        for mod in self.encoder_stem_prot:
            s_prot = mod(s_prot)
            if isinstance(mod, EncoderBlock) and mod.downsample:
                encoder_stem_prot_s.append(s_prot)

        s_prot_all.extend(encoder_stem_prot_s)

        # combine property branch and cumulative density branch
        s_prot = torch.cat([s_prot, s_prot2], dim=1)

        # run encoder main prot
        encoder_main_prot_s = []
        for mi, mod in enumerate(self.encoder_main_prot):
            next_mod = self.encoder_main_prot[mi + 1] if mi < len(self.encoder_main_prot) - 1 else None
            s_prot = mod(s_prot)
            if self.combine_prot_dec == 'scale':
                if mi == len(self.encoder_main_prot) - 1 or getattr(mod, 'downsample', False):
                    encoder_main_prot_s.append(s_prot)
            elif self.combine_prot_dec == 'group' and isinstance(mod, EncoderModule): # not getattr(next_mod, 'downsample', False):
                encoder_main_prot_s.append(s_prot)
        s_prot_all.extend(encoder_main_prot_s)

        # reverse for running decoder
        s_prot_all.reverse()

        ####
        
        if plist is not None:
            dist = NormalDist(plist[0][0].cuda(), plist[0][1].cuda(), temp=temps[di])
        else:
            dist = NormalDist(mu=torch.zeros(z0_shape).cuda(), log_sigma=torch.zeros(z0_shape).cuda(), temp=temps[di])
        
        z_latent, _ = dist.sample()
        if zlist is not None:
            z_sample = zlist[0]
            z_latent = torch.lerp(z_sample, z_latent, zbeta[0])

        di = 0
        s = self.prior_dec_input0.unsqueeze(0)
        n_batch = z_latent.size(0)
        s = s.expand(n_batch, -1, -1, -1, -1)

        pi = 0
        for mi, mod in enumerate(self.decoder_main):
            if isinstance(mod, DecoderMixBlockProt):
                try:
                    s = mod(s, s_prot_all[pi])
                except Exception as e:
                    print(mi, pi, s.shape, s_prot_all[pi].shape)
                    print('pi-1:', s_prot_all[pi-1].shape, '\tpi+1:', s_prot_all[pi+1].shape)
                    raise e
                pi += 1
                
            elif isinstance(mod, DecoderMixBlock):
                if di > 0:                    
                    # decoder prior dist params
                    dist_params = self.decoder_sampler[di - 1](s)
                    mu_p, log_sigma_p = torch.chunk(dist_params, 2, dim=1)

                    idxs = (spatial_idxs * mu_p.shape[-1]).round().astype(int) if spatial_idxs is not None else None
                    zb = zbeta[di] if len(zbeta) == n_latents else zbeta[up_level]
                    temp = temps[di] if len(temps) == n_latents else temps[up_level]

                    if plist is not None:
                        mu_sample, log_sig_sample = plist[di]
                        mu_sample, log_sig_sample =  mu_sample.cuda(), log_sig_sample.cuda()
                        if spatial_idxs is not None:
                            mu_new, log_sigma_new = mu_sample.clone(), log_sig_sample.clone()
                            mu_new[:,:, idxs[0][0]:idxs[0][1], idxs[1][0]:idxs[1][1], idxs[2][0]:idxs[2][1]] = torch.lerp(mu_sample, mu_p, zb)[:,:, idxs[0][0]:idxs[0][1], idxs[1][0]:idxs[1][1], idxs[2][0]:idxs[2][1]]
                            log_sigma_new[:,:, idxs[0][0]:idxs[0][1], idxs[1][0]:idxs[1][1], idxs[2][0]:idxs[2][1]] = torch.lerp(log_sig_sample, log_sigma_p, zb)[:,:, idxs[0][0]:idxs[0][1], idxs[1][0]:idxs[1][1], idxs[2][0]:idxs[2][1]]
                            mu_p, log_sigma_p = mu_new, log_sigma_new
                        else:
                            mu_p = torch.lerp(mu_sample, mu_p, zb)
                            log_sigma_p = torch.lerp(log_sig_sample, log_sigma_p, zb)
                    
                    
                    dist = NormalDist(mu_p, log_sigma_p, temp=temp)
                    z_latent, _ = dist.sample()
                    
                    ## interpolate mu relative to z from sample
                    if zlist is not None:
                        z_sample = zlist[di]
                        if spatial_idxs is not None:
                            if invert_spatial:

                                z_new = torch.lerp(z_sample, z_latent, zb)
                                z_new[:,:, idxs[0][0]:idxs[0][1], idxs[1][0]:idxs[1][1], idxs[2][0]:idxs[2][1]] = z_sample[:,:, idxs[0][0]:idxs[0][1], idxs[1][0]:idxs[1][1], idxs[2][0]:idxs[2][1]].clone()
                            else:
                                z_new = z_sample.clone()
                                z_new[:,:, idxs[0][0]:idxs[0][1], idxs[1][0]:idxs[1][1], idxs[2][0]:idxs[2][1]] = torch.lerp(z_sample, z_latent, zb)[:,:, idxs[0][0]:idxs[0][1], idxs[1][0]:idxs[1][1], idxs[2][0]:idxs[2][1]]
                            
                            z_latent = z_new
                        else:
                            z_latent = torch.lerp(z_sample, z_latent, zb)

                s = mod(s, z_latent)
                di += 1
            else:
                s = mod(s)
                if getattr(mod, 'upsample', False):
                    up_level += 1


        ### res2 output
        s2 = s
        for mod in self.decoder_stem2:
            s2 = mod(s2)
        out2 = self.output_module2(s2)
        ###


        # run decoder stem
        for mod in self.decoder_stem:
            if isinstance(mod, EncoderMixBlockProt):
                s = mod(s, s_prot_all[pi])
                pi += 1
            else:
                s = mod(s)

        outputs = (self.output_module(s), out2)
        
        return outputs