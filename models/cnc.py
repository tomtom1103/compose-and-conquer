import os
import sys
if './' not in sys.path:
	sys.path.append('./')
import einops
import torch
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from models.hack import print_all_children, count_params

class CnC(LatentDiffusion):
    def __init__(self, mode, local_control_config=None, global_control_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['local', 'global', 'cnc']
        self.mode = mode
        if self.mode in ['local', 'cnc']:
            self.local_adapter = instantiate_from_config(local_control_config)
            self.local_control_scales = [1.0] * 13
        if self.mode in ['global', 'cnc']:
            self.global_adapter = instantiate_from_config(global_control_config)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        if self.mode == 'local':
            bg_depth = batch['bg_depth']
            bg_depth = einops.rearrange(bg_depth, 'b h w c -> b c h w')
            bg_depth = bg_depth.to(self.device).to(memory_format=torch.contiguous_format).float()

            fg_depth = batch['fg_depth']
            fg_depth = einops.rearrange(fg_depth, 'b h w c -> b c h w')
            fg_depth = fg_depth.to(self.device).to(memory_format=torch.contiguous_format).float()

            bg_emb = torch.zeros(1,1).to(self.device).to(memory_format=torch.contiguous_format).float()
            fg_emb = torch.zeros(1,1).to(self.device).to(memory_format=torch.contiguous_format).float()

            mask = torch.zeros(1,1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float()

        elif self.mode == 'global':
            bg_emb = batch['bg_emb']
            bg_emb = bg_emb.to(self.device).to(memory_format=torch.contiguous_format).float()

            fg_emb = batch['fg_emb']
            fg_emb = fg_emb.to(self.device).to(memory_format=torch.contiguous_format).float()

            mask = batch['mask']
            mask = mask.to(self.device).to(memory_format=torch.contiguous_format).float()

            bg_depth = torch.zeros(1,1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float()
            fg_depth = torch.zeros(1,1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float()

        elif self.mode == 'cnc':
            bg_depth = batch['bg_depth']
            bg_depth = einops.rearrange(bg_depth, 'b h w c -> b c h w')
            bg_depth = bg_depth.to(self.device).to(memory_format=torch.contiguous_format).float()

            fg_depth = batch['fg_depth']
            fg_depth = einops.rearrange(fg_depth, 'b h w c -> b c h w')
            fg_depth = fg_depth.to(self.device).to(memory_format=torch.contiguous_format).float()

            bg_emb = batch['bg_emb']
            bg_emb = bg_emb.to(self.device).to(memory_format=torch.contiguous_format).float()

            fg_emb = batch['fg_emb']
            fg_emb = fg_emb.to(self.device).to(memory_format=torch.contiguous_format).float()

            mask = batch['mask']
            mask = mask.to(self.device).to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c],
                    bg_depth=[bg_depth],
                    fg_depth=[fg_depth],
                    bg_emb=[bg_emb],
                    fg_emb=[fg_emb],
                    mask=[mask])

    def apply_model(self, x_noisy, t, cond, global_strength=[1,1], *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if self.mode in ['global']:
            assert cond['bg_emb'][0] != None
            assert cond['fg_emb'][0] != None
            assert cond['mask'][0] != None

            bg_emb = cond["bg_emb"][0]
            fg_emb = cond["fg_emb"][0]
            mask = cond["mask"][0]
            global_control = self.global_adapter(bg_emb, fg_emb)
            fg_tokens = global_control[:,:4,:] * global_strength[0]
            bg_tokens = global_control[:,4:,:] * global_strength[1]
            global_control = torch.cat([fg_tokens, bg_tokens], dim=1)
            cond_txt = torch.cat([cond_txt, global_control], dim=1)

        if self.mode in ['local']:
            assert cond['bg_depth'][0] != None
            assert cond['fg_depth'][0] != None
            bg_depth = torch.cat(cond['bg_depth'], 1)
            fg_depth = torch.cat(cond['fg_depth'], 1)
            local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, bg_depth=bg_depth, fg_depth=fg_depth)
            local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]
        
        if self.mode in ['cnc']:
            assert cond['bg_emb'][0] != None
            assert cond['fg_emb'][0] != None
            assert cond['mask'][0] != None
            assert cond['bg_depth'][0] != None
            assert cond['fg_depth'][0] != None

            bg_emb = cond["bg_emb"][0]
            fg_emb = cond["fg_emb"][0]
            mask = cond["mask"][0]
            global_control = self.global_adapter(bg_emb, fg_emb)
            fg_tokens = global_control[:,:4,:] * global_strength[0]
            bg_tokens = global_control[:,4:,:] * global_strength[1]
            global_control = torch.cat([fg_tokens, bg_tokens], dim=1)
            cond_txt = torch.cat([cond_txt, global_control], dim=1)

            bg_depth = torch.cat(cond['bg_depth'], 1)
            fg_depth = torch.cat(cond['fg_depth'], 1)
            local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, bg_depth=bg_depth, fg_depth=fg_depth, mask=mask)
            local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]

        
        if self.mode == 'global':
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, mask=mask)
        elif self.mode == 'local':
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, local_control=local_control)
        else:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, local_control=local_control, mask=mask)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False,
                   unconditional_guidance_scale=9.0, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        bg_depth = c["bg_depth"][0][:N]
        fg_depth = c["fg_depth"][0][:N]
        bg_emb = c["bg_emb"][0][:N]
        fg_emb = c["fg_emb"][0][:N]
        mask = c["mask"][0][:N]
        c = c["c_crossattn"][0][:N]
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        if self.mode in ['local', 'cnc']:
            log["bg_depth"] = bg_depth
            log["fg_depth"] = fg_depth

        if self.mode in ['global', 'cnc']:
            # log["bg_color"] = bg_color
            # log["fg_color"] = fg_color
            pass
        # log["prompt"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if sample:
            samples, z_denoise_row = self.sample_log(cond={"bg_depth": [bg_depth],
                                                           "fg_depth": [fg_depth],
                                                           "bg_emb": [bg_emb],
                                                           "fg_emb": [fg_emb],
                                                           "c_crossattn": [c],
                                                           "mask": [mask]
                                                           },
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_bg_depth = bg_depth
            uc_fg_depth = fg_depth
            # uc_global = torch.zeros_like(c_global)
            uc_full = {"bg_depth": [uc_bg_depth],
                        "fg_depth": [uc_fg_depth],
                        "bg_emb": [bg_emb],
                        "fg_emb": [fg_emb],
                        "c_crossattn": [uc_cross],
                        "mask": [mask]}
            samples_cfg, _ = self.sample_log(cond={"bg_depth": [bg_depth],
                                                    "fg_depth": [fg_depth],
                                                    "bg_emb": [bg_emb],
                                                    "fg_emb": [fg_emb],
                                                    "c_crossattn": [c],
                                                    "mask": [mask]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        if self.mode == 'global':
            h, w = 512, 512
        else:
            _, _, h, w = cond["bg_depth"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, global_strength=[1,1], **kwargs)
        return samples, intermediates

    def configure_optimizers(self): # 여기서 어떤 파라미터 학습하고 어떤걸 freeze 할지
        lr = self.learning_rate
        params = []
        if self.mode in ['local', 'cnc']:
            params += list(self.local_adapter.parameters())
        if self.mode in ['global', 'cnc']:
            params += list(self.global_adapter.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            if self.mode in ['local', 'cnc']:
                self.local_adapter = self.local_adapter.cuda()
            if self.mode in ['global', 'cnc']:
                self.global_adapter = self.global_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            if self.mode in ['local', 'cnc']:
                self.local_adapter = self.local_adapter.cpu()
            if self.mode in ['global', 'cnc']:
                self.global_adapter = self.global_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from models.hack import disable_verbosity, count_params, print_all_children
    from torch.utils.data import DataLoader
    disable_verbosity()
    