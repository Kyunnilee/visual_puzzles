import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import create_directory


# Set up logging + Disable annoying HTTPX logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info("Starting the evaluation process...")

    # Resolve all interpolations
    OmegaConf.resolve(cfg)
    # Convert the fully resolved configuration to YAML
    yaml_cfg = OmegaConf.to_yaml(cfg)
    logging.info(f"Using configuration:\n{yaml_cfg}")
    create_directory(cfg.basic.output_dir)

    solver = hydra.utils.instantiate(cfg.solver)
    logging.info("Solver instantiated successfully.")
    # Set the start to zero and end to the length of the dataset if not specified
    # solver.run_batch(cfg.basic.output_dir, start_idx=0, end_idx=5)
    solver.run_batch(
        cfg.basic.output_dir,
        use_icl=cfg.basic.use_icl,
        use_skill_prompt=cfg.basic.use_skill_prompt,
        use_caption=cfg.basic.use_caption,
    )
    logging.info("Batch processing completed.")
    logging.info(f"Running evaluation for ({cfg.solver.name},{cfg.basic.mode}) -- TODO")
    # run_eval(sub_dir_name, cfg.basic.mode)


if __name__ == "__main__":
    main()
