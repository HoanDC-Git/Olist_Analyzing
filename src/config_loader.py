import os
import yaml

def load_config():
    """Loads configuration from config.yaml and resolves relative paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Resolve all directory paths relative to the project root
    for key, path_val in config["paths"].items():
        config["paths"][key] = os.path.join(base_dir, path_val)
        os.makedirs(config["paths"][key], exist_ok=True)
        
    return config

if __name__ == "__main__":
    cfg = load_config()
    print("Resolved Paths:")
    for k, v in cfg["paths"].items():
        print(f"  {k}: {v}")
