import argparse, os
from pathlib import Path

from Akordio_Core.Classes.NetConfig import load_config

from Testers.BaseTester import BaseTester
from Testers.CRFTester import CRFTester
from Testers.LogCRFTester import LogCRFTester


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ModelTester", description="Program for testing chord recognition models")
    parser.add_argument("-c", "--crf", action="store_true", help="Whether to test the model with its CRF layer")
    parser.add_argument("-t", "--test", action="store_true", help="Run final evaluation on test set")
    parser.add_argument("-m", "--model", required=True, type=str, help="Name of the model to test")
    parser.add_argument("-f", "--fold", required=True, type=str, help="Number of the validation fold")
    args = parser.parse_args()

    # Params
    crf_enabled: bool = args.crf
    test_enabled: bool = args.test
    model_name: str = args.model
    fold: str = args.fold

    # Find model dir
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_dir = os.path.join(project_root, "Models", model_name, fold)
    if not os.path.exists(model_dir):
        parser.error("Specified model and fold for it have not been found!")

    # Config
    config_path = os.path.join(model_dir, "config.yaml")
    config = load_config(config_path)

    # Logistic
    log_enabled = (config.train.model_type == "LOG")

    # Class choice
    if not crf_enabled:         # Non CRF
        if not log_enabled:
            tester = BaseTester(config)
        else:
            raise NotImplementedError()
    else:                       # CRF
        if not log_enabled:
            tester = CRFTester(config)
        else:
            tester = LogCRFTester(config)  
    
    # Test
    tester.test(test_enabled)