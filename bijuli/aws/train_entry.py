"""SageMaker training entry point (script mode, run on a managed spot instance).

Reads the parquet written by `build_dataset` from the `train` channel, fits the quantile forecaster
with conformal calibration, and writes model.joblib plus a model card to /opt/ml/model. SageMaker
uploads that directory to S3 as model.tar.gz; the next state copies it to models/<version>/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd

from bijuli.models import QuantileForecaster


def main() -> None:
    train_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    frame = pd.concat(pd.read_parquet(p) for p in sorted(train_dir.glob("*.parquet")))
    model = QuantileForecaster().fit(frame)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    card = {**model.meta, "holdout_mape": model.holdout_mape, "conformal_margin": model.margin}
    (model_dir / "card.json").write_text(json.dumps(card, default=str))
    print(json.dumps({"holdout_mape": model.holdout_mape}))  # parsed by the metric definition


if __name__ == "__main__":
    main()
