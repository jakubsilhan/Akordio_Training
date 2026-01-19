from Testers.LogCRFTester import LogCRFTester
from Akordio_Core.Classes.NetConfig import load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    tester = LogCRFTester(config)
    tester.test()