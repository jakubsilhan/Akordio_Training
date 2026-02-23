from argparse import ArgumentParser

from Akordio_Core.Classes.NetConfig import load_config

from Trainers.BaseTrainer import BaseTrainer
from Trainers.MultiTrainer import MultiTrainer
from Trainers.LogisticTrainer import LogisticTrainer
from Trainers.CRFTrainer import CRFTrainer
from Trainers.LogCRFTrainer import LogCRFTrainer

if __name__=="__main__":
    parser = ArgumentParser(prog="ModelTrainer", description="Program for training chord recognition models")
    parser.add_argument("-c", "--crf", action="store_true", help="Whether to train the CRF part of the model")
    parser.add_argument("-f", "--final", action="store_true", help="Whether to train the final model")
    parser.add_argument("-m", "--multitask", action="store_true", help="Whether to use multitask training")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs for the final run")
    args = parser.parse_args()
    
    # Params
    crf_enabled = args.crf
    final_enabled = args.final
    multi_enabled = args.multitask

    if args.final and args.epochs is None:
        parser.error("--final requires --epochs [count]")

    # Config
    config = load_config("config.yaml")

    # Logistic
    log_enabled = (config.train.model_type == "LOG")

    if multi_enabled and (crf_enabled or log_enabled):
        parser.error("Multitask training is not supported with CRF or Logistic model types.")
        
    # Class choice
    if not crf_enabled:     # Non-CRF
        if not log_enabled:
            if not multi_enabled:
                trainer = BaseTrainer(config)
            else:
                trainer = MultiTrainer(config)
        else:
            trainer = LogisticTrainer(config)
    else:                   # CRF
        if not log_enabled:
            trainer = CRFTrainer(config)
        else:
            trainer = LogCRFTrainer(config)

    # Training choice
    if not final_enabled:   # Train - normal
        trainer.train()
    else:                   # Train - final
        trainer.train_final(int(args.epochs))