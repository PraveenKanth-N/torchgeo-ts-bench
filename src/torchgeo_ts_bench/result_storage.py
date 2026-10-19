"""Result storage and artifact persistence module for TorchGeo-TS-Bench."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .evaluation import EvaluationResults


class BenchmarkResultStorage:
    """Manages CSV, JSON, and YAML result files in the outputs directory."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "results.csv")
        self.metrics_json_path = os.path.join(self.output_dir, "metrics.json")
        self.config_yaml_path = os.path.join(self.output_dir, "experiment_config.yaml")

    def save_experiment_config(self, config_dict: Dict[str, Any]) -> str:
        """Save YAML configuration file for exact reproducibility."""
        with open(self.config_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        return self.config_yaml_path

    def save_metrics_json(self, full_metrics: Dict[str, Any]) -> str:
        """Save detailed hierarchical metrics JSON file."""
        with open(self.metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(full_metrics, f, indent=2, default=str)
        return self.metrics_json_path

    def append_or_write_results_csv(
        self, records: List[Dict[str, Any]], overwrite: bool = False
    ) -> str:
        """Write or append tabular evaluation records to results.csv."""
        new_df = pd.DataFrame(records)
        if os.path.exists(self.csv_path) and not overwrite:
            existing_df = pd.read_csv(self.csv_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove exact duplicates if re-run with same run_id/model
            if "run_id" in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=["run_id", "model_name"], keep="last")
            combined_df.to_csv(self.csv_path, index=False)
        else:
            new_df.to_csv(self.csv_path, index=False)
        return self.csv_path

    def load_results_df(self) -> pd.DataFrame:
        """Load tabular results as a pandas DataFrame."""
        if os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        return pd.DataFrame()

    def load_metrics_json(self) -> Dict[str, Any]:
        """Load full metrics JSON file."""
        if os.path.exists(self.metrics_json_path):
            with open(self.metrics_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
