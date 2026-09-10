from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "reactor_params.yaml"


def load_params(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        params = yaml.safe_load(f)

    numeric_blocks = [
        "certain_params", "nominal_feed", "uncertain_params",
        "nmpc", "economics", "rl", "measurement", "estimator",
    ]
    for block in numeric_blocks:
        if block not in params:
            continue
        for k, v in params[block].items():
            if isinstance(v, str):
                params[block][k] = float(v)
            elif isinstance(v, list):
                params[block][k] = [float(x) for x in v]

    for block in ("state_bounds", "input_bounds"):
        for var, bounds in params.get(block, {}).items():
            for k, v in bounds.items():
                if isinstance(v, str):
                    bounds[k] = float(v)

    for k, v in params.get("initial_state", {}).items():
        if isinstance(v, str):
            params["initial_state"][k] = float(v)

    return params


if __name__ == "__main__":
    import json
    print(json.dumps(load_params(), indent=2))
