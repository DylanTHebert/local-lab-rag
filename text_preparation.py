import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="terms-service.yaml")
def main(cfg : DictConfig) -> None:
    """Entry point for textual data prep

    ingests, parse, batch, wrap and transfer to home server"""
    print(OmegaConf.to_yaml(cfg))
    embedding = hydra.utils.instantiate(cfg.dataprep.embedding)



if __name__ == "__main__":
    main()