from Testers.CRFTester import CRFTester
from Akordio_Core.Classes.NetConfig import load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    tester = CRFTester(config)
    tester.test()