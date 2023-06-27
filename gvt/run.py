import os
import copy
import torch
from collections import OrderedDict
import pytorch_lightning as pl

from gvt.config import ex
from gvt.modules import GVT
from gvt.datamodules.multitask_datamodule import MTDataModule

from pytorch_lightning.plugins import environments as pl_env
from pytorch_lightning.utilities.distributed import rank_zero_info

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class OMPIClusterEnvironment(pl_env.ClusterEnvironment):
    def __init__(self):
        super().__init__()

    @property
    def creates_processes_externally(self):
        return True

    def world_size(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def set_world_size(self, size: int):
        pass

    def global_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    def set_global_rank(self, rank: int):
        pass

    def local_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    def node_rank(self) -> int:
        if "NODE_RANK" in os.environ:
            return int(os.environ["NODE_RANK"])
        else:
            return 0

    def master_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])


def get_cluster_plugin(num_gpus=1, num_nodes=1):
    if num_nodes > 1 or (
        num_nodes == 1 and "OMPI_COMM_WORLD_SIZE" in os.environ
    ):
        rank_zero_info("ClusterPlugin: using OMPI Cluster Environment")
        return OMPIClusterEnvironment()
    if num_gpus >= 1:
        rank_zero_info("ClusterPlugin: using Lightning Cluster Environment")
        return pl_env.LightningEnvironment()
    return None



@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

   
    model = GVT(config=_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        verbose=True,
        monitor="val/the_metric",
        mode="min",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    resume_ckpt = _config['load_path']
    if _config["test_only"] and resume_ckpt is not None:
        state_dict = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("load state dict from:", resume_ckpt)

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        strategy="deepspeed_stage_2",
        benchmark=True,
        deterministic=False, 
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        replace_sampler_ddp=False,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        resume_from_checkpoint=resume_ckpt,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        model.eval()
        with torch.no_grad():
            trainer.test(model, datamodule=dm)
