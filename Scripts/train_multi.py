from Trainers.MultiTrainer import MultiTrainer
from Akordio_Core.Classes.NetConfig import load_config

if __name__=="__main__":
    config = load_config("config.yaml")
    trainer = MultiTrainer(config)
    trainer.train()