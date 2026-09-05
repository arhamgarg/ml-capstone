from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATASETS = {
    "regression": 294,
    "classification": 350,
    "clustering": 292,
}


def main() -> None:
    output_dir = Path(__file__).parent
    for name, dataset_id in DATASETS.items():
        dataset = fetch_ucirepo(id=dataset_id)
        frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        frame.to_csv(output_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
