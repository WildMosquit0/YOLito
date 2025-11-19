import yaml
import os
from typing import Dict

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def update_yaml(config, yaml_path,field):
    """Update the YAML file with the latest changes in the config."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update the 'input_csv' key with the new path
    data[field] = config[field]
    
    # Write the updated data back to the YAML file
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(data, file)




def export_config(conf_yaml_path):

    config_to_export = load_config(conf_yaml_path)
    output_dir = config_to_export["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    task = "infer" if "infer" in os.path.basename(conf_yaml_path) else "analyze"
    config_path = os.path.join(output_dir, f"{task}_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_to_export, f, default_flow_style=False)

    print(f"Exported config to {config_path}")


def update_analyze_config(config: Dict) -> None:
    

    analyze_path = config.replace("infer",'analyze')
    if not os.path.exists(analyze_path):
        print(f"Analyze config not found at: {analyze_path}")
        return

    analyze_conf = load_config(analyze_path)
    infer_conf = load_config(config)
    output_dir = infer_conf["output_dir"]
    analyze_conf["input_csv"] = os.path.join(output_dir, "results.csv")
    analyze_conf["output_dir"] = output_dir

    if "heatmap" in analyze_conf:
        analyze_conf["heatmap"]["image_path"] = os.path.join(output_dir, "frames")

    with open(analyze_path, "w") as f:
        yaml.dump(analyze_conf, f, default_flow_style=False)

    print(f"Updated analyze config at {analyze_path}")
