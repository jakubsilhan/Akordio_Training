from Trainers.LogCRFTrainer import LogCRFTrainer
from Akordio_Core.Classes.NetConfig import load_config
if __name__=="__main__":
    config = load_config("config.yaml")
    trainer = LogCRFTrainer(config)
    trainer.train()