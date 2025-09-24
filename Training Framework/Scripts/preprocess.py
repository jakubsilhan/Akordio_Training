from Core.config import Config, load_config
from Core.preprocessor import Preprocessor

if __name__ == "__main__":
    config = load_config("config.yaml")
    preprocessing = Preprocessor(config)
    preprocessing.process_all_data()