import pandas as pd
import numpy as np
import random
from pathlib import Path
import argparse
from typing import Literal

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Predefined mock domains for CAR structure
SCFV_LIBRARY = ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQ", "MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFY"]
HINGES = ["GGGGS", "APAPT", "GPGPG"]
TRANSMEM = ["LVVIVLAVLVAL"]
INTRACELL = ["CD3Z", "4-1BB", "CD28"]

TARGET_ANTIGENS = ["CD19", "BCMA", "HER2", "EGFR"]

CLINICAL_OUTCOMES = ["effective", "ineffective"]
CYTOKINE_RELEASE = ["low", "high"]


def random_seq(length=30):
    return ''.join(random.choices(AMINO_ACIDS, k=length))

def build_cart(seq_id):
    return (
        random.choice(SCFV_LIBRARY) +
        random.choice(HINGES) +
        random.choice(TRANSMEM) +
        random.choice(INTRACELL)
    )

def generate_dummy_cart_dataset(n=100, output_dir="cart_dummy_data"):
    Path(output_dir).mkdir(exist_ok=True)

    data = []
    multilabel_targets = []
    regression_targets = []
    multiclass_targets = []
    clinical_effectiveness = []
    cytokine_labels = []
    paired_antigens = []
    family_ids = []

    for i in range(n):
        seq = build_cart(i)
        antigen = random.choice(TARGET_ANTIGENS)
        fam = antigen  # Group by antigen family for grouping experiments

        # General
        multilabel_target = np.random.randint(0, 2, size=5).tolist()  # e.g., GO terms
        regression_target = round(random.uniform(0.0, 1.0), 3)        # e.g., affinity
        multiclass_target = random.randint(0, 4)                      # e.g., function class

        # CAR-specific labels
        is_effective = random.choice(CLINICAL_OUTCOMES)
        cytokine = random.choice(CYTOKINE_RELEASE)

        data.append({
            "Sequence": seq,
            "Antigen": antigen,
            "FamilyID": fam,
            "Index": i,
        })

        multilabel_targets.append(multilabel_target)
        regression_targets.append(regression_target)
        multiclass_targets.append(multiclass_target)
        clinical_effectiveness.append(1 if is_effective == "effective" else 0)
        cytokine_labels.append(1 if cytokine == "high" else 0)
        paired_antigens.append(antigen)
        family_ids.append(fam)

    # Save sequence metadata
    df = pd.DataFrame(data)
    df.to_parquet(Path(output_dir) / "data.parquet", index=False)

    # Save target matrices
    np.save(Path(output_dir) / "multilabel_targets.npy", np.array(multilabel_targets))
    np.save(Path(output_dir) / "regression_targets.npy", np.array(regression_targets))
    np.save(Path(output_dir) / "multiclass_targets.npy", np.array(multiclass_targets))
    np.save(Path(output_dir) / "clinical_effective.npy", np.array(clinical_effectiveness))
    np.save(Path(output_dir) / "cytokine_levels.npy", np.array(cytokine_labels))
    np.save(Path(output_dir) / "antigen_names.npy", np.array(paired_antigens))
    np.save(Path(output_dir) / "family_ids.npy", np.array(family_ids))

    print(f"✅ Generated dummy CAR-T dataset with {n} samples in '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of dummy sequences to generate")
    parser.add_argument("--out", type=str, default="cart_dummy_data", help="Output directory")
    args = parser.parse_args()
    generate_dummy_cart_dataset(n=args.n, output_dir=args.out)
