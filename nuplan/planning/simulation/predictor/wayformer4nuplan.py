import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluation(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)

    val_set = build_dataset(cfg, val=True)

    eval_batch_size = cfg.method['eval_batch_size']

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)

    trainer = pl.Trainer(
        inference_mode=True,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name),
        devices=1,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
    )

    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    evaluation()

@hydra.main(version_base=None, config_path="configs", config_name="config")
class Wayformer4Nuplan:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(self.cfg.seed)
        OmegaConf.set_struct(self.cfg, False)
        self.cfg = OmegaConf.merge(self.cfg, self.cfg.method)
        cfg['eval'] = True

        self.model = build_model(self.cfg)
        self.ckpt_path=self.cfg.ckpt_path

        self.trainer = pl.Trainer(
            inference_mode=True,
            logger=None,
            devices=1,
            accelerator="cpu" if cfg.debug else "gpu",
            profiler="simple",
        )

    def wayformer_inference(self,data):
        # Load the model and perform inference
        inference_data = DataLoader()
        predictions = self.trainer.prdict(model=self.model, dataloaders=inference_data, ckpt_path=self.ckpt_path,return_predictions=True)

        return predictions
