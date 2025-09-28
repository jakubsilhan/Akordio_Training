from core.net_config import Config, load_config
from core.preprocessor import Preprocessor

if __name__ == "__main__":
    config = load_config("config.yaml")
    preprocessing = Preprocessor(config)
    preprocessing.process_all_data()