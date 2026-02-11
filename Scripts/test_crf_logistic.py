import argparse

from Testers.LogCRFTester import LogCRFTester
from Akordio_Core.Classes.NetConfig import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Akordio Training Framework")
    parser.add_argument("--test", action="store_true", help="Run final evaluation on test set")
    args = parser.parse_args()

    config = load_config("config.yaml")
    tester = LogCRFTester(config)
    tester.test(args.test)