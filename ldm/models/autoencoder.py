import torch
import pytorch_lightning as pl
from torch import Tensor
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

if __name__ == "__main__":

    config_path: str = "/data/davidn/PhXD/GeometricDeepLearning/latent-diffusion/configs/autoencoder/autoencoder_kl_8x8x64.yaml"
    # config_path: str = "/data/davidn/PhXD/GeometricDeepLearning/latent-diffusion/configs/autoencoder/autoencoder_kl_64x64x3.yaml"
    # config_path: str = "/data/davidn/PhXD/GeometricDeepLearning/latent-diffusion/configs/autoencoder/autoencoder_kl_16x16x16.yaml"
    # config_path: str = "/data/davidn/PhXD/GeometricDeepLearning/latent-diffusion/configs/autoencoder/autoencoder_kl_custom.yaml"

    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    # lossconfig = OmegaConf.create({'target': 'ldm.losses.AutoencoderLoss', 'params': {'disc_start': 10000, 'feature_loss_weight': 1.0, 'adversarial_loss_weight': 1.0}})
    model = AutoencoderKL(**config['model']['params'])
    state_dict: dict = torch.load("./model_kl-f32.ckpt")['state_dict']
    # use everything except for keys that start with "loss" or "quant"

    state_dict = state_dict.copy()
    # state_dict = {k: v for k, v in state_dict.items() if not "loss" in k and not "quant" in k}

    # print(f"{state_dict.keys()=}")
    model.load_state_dict(state_dict, strict=False, )
    output: DiagonalGaussianDistribution = model.encode(torch.randn(1, 3, 512, 512))
    print(f"{output.sample().shape=}")

    # load in an image and encode/decode
    from PIL import Image
    import numpy as np
    img = Image.open("/data/davidn/PhXD/GeometricDeepLearning/latent-diffusion/lie.png").convert("RGB")
    img = img.resize((512, 512), Image.LANCZOS)
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    print(f"{img.shape=}")
    output: DiagonalGaussianDistribution = model.encode(img) 
    print(f"{output.sample().shape=}")

    out_smpl: Tensor = output.sample()
    rec: Tensor = model.decode(out_smpl)
    print(f"{rec.shape=}")
    rec = ((rec + 1.0) * 127.5).clamp(0, 255)
    rec = rec.detach().cpu().numpy().astype(np.uint8)
    rec = rec.squeeze().transpose(1, 2, 0)
    rec = Image.fromarray(rec)

    # plot side by side
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(((img.squeeze().permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8))
    axs[0].set_title("Original")
    axs[1].imshow(rec)
    axs[1].set_title("Reconstruction")
    plt.savefig("reconstruction.png")