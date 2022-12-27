import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from models.dataloader import Dataloader
from models.metrics import Metric
from models.model import Model
from utils import make_file_name, print_config, print_msg, setdir


def train(conf, version, is_wandb: bool = True, is_scheduler: bool = True):
    print_config(conf)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = setdir(conf.data_dir, conf.save_dir_name, reset=False)

    # load dataset & dataloader
    train_dataloader = Dataloader(conf, mode="train")
    metric_object = Metric(conf, mode="validation")
    # load model
    model = Model(conf, device, metric_object=metric_object, is_scheduler=is_scheduler)
    model.to(device)

    # set checkpoints
    # ckpt_dir_name = f"ckpt_{conf.run_name.split('/')[-1]}"
    # ckpt_dirpath = setdir(conf.data_dir, ckpt_dir_name, reset=False)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filename="{epoch:02d}_{micro_f1_score:.3f}",
    #     save_top_k=3,
    #     dirpath=ckpt_dirpath,
    #     monitor="micro_f1_score",
    #     mode="max",
    # )

    # declare callback function of lr monitoring
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    model_name = conf.model_name.replace("/", "_")
    if is_wandb:
        wandb_logger = WandbLogger(
            project=conf.project,
            entity="boost2end",
            name=conf.run_name,
            save_dir=os.path.join(conf.data_dir),
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=conf.max_epoch,
            log_every_n_steps=1,
            logger=wandb_logger,
            precision=16,
            callbacks=[lr_monitor],  # , checkpoint_callback],
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=conf.max_epoch,
            log_every_n_steps=1,
            precision=16,
            callbacks=[lr_monitor],  # , checkpoint_callback],
        )

    # Train part
    print_msg("학습을 시작합니다...", "INFO")
    trainer.fit(model=model, datamodule=train_dataloader)
    print_msg("학습이 종료되었습니다...", "INFO")

    # Validation part
    print_msg("최종 모델 검증을 시작합니다...", "INFO")
    trainer.test(model=model, datamodule=train_dataloader)
    print_msg("최종 모델 검증이 종료되었습니다...", "INFO")

    # 학습이 완료된 모델의 state dict 저장
    print_msg("마지막 모델을 저장합니다...", "INFO")
    file_name = make_file_name(model_name, format="pt", version=version)
    model_path = os.path.join(save_path, file_name)
    torch.save(model.state_dict(), model_path)
