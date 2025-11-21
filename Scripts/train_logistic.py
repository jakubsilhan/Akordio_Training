from Trainers.LogisticTrainer import LogisticTrainer
from Akordio_Core.Classes.NetConfig import load_config


if __name__=="__main__":
    config = load_config("config.yaml")
    trainer = LogisticTrainer(config)
    trainer.train()