from typing import Optional

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train_ppo_agent.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    pass


if __name__ == "__main__":
    main()
