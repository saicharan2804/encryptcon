import hydra
from omegaconf import DictConfig
from train_vae import VaeTrainer
from train_llm import Trainer

TRAINER = {"llm": Trainer, "vae": VaeTrainer}


@hydra.main(version_base=None, config_path="configs", config_name="llm")
def train(cfg: DictConfig):
    """Function to run the train pipeline.

    Provides a simple interface to train any model [VAE|LLM]

    Args:
        cfg : DictConfig
            Config file containing all experiments such as training hyperparameters,
            model configs, etc.
    """

    trainer = TRAINER[cfg.mode](cfg)
    trainer.train()

if __name__ == "__main__":
    train()
