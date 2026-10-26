"""A small model registry with the same lifecycle as a SageMaker Model Registry model package group:

    candidate -> (offline gate) -> shadow -> (live gate) -> champion -> archived
                         \\-> rejected           \\-> rejected       \\-> rolled_back

Each version stores the model, its training-data hash and window, its offline metrics and the
feature reference distributions the drift monitor compares against. The champion pointer is a single
value (an SSM parameter on AWS), so promotion and rollback are one atomic write and an audit event.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib

STATUSES = {"candidate", "shadow", "champion", "archived", "rejected", "rolled_back"}


@dataclass
class ModelCard:
    version: str
    created: str  # issue day the model was trained on (replay clock), ISO date
    status: str
    reason: str
    train_start: str
    train_end: str
    data_sha256: str
    offline: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)  # feature -> {"edges": [...], "freq": [...]}
    params: dict = field(default_factory=dict)


class Registry:
    def __init__(self, root: str | Path, persist_models: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.persist_models = persist_models
        self.index_path = self.root / "registry.json"
        self.index = {"champion": None, "previous_champion": None, "cards": {}, "events": []}
        self._models: dict[str, object] = {}
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text())

    # -- writes -----------------------------------------------------------------------------
    def register(self, model, card: ModelCard) -> str:
        if card.status not in STATUSES:
            raise ValueError(card.status)
        self._models[card.version] = model
        if self.persist_models:
            d = self.root / card.version
            d.mkdir(exist_ok=True)
            joblib.dump(model, d / "model.joblib")
        self.index["cards"][card.version] = asdict(card)
        self._event(card.created, card.version, "registered", card.reason)
        self._save()
        return card.version

    def set_status(self, version: str, status: str, when: str, reason: str) -> None:
        if status not in STATUSES:
            raise ValueError(status)
        self.index["cards"][version]["status"] = status
        self._event(when, version, status, reason)
        self._save()

    def promote(self, version: str, when: str, reason: str) -> None:
        if version not in self.index["cards"]:
            raise KeyError(f"unknown model version {version!r}")
        old = self.index["champion"]
        if version == old:
            # Re-promoting the champion would archive it and make it its own rollback target,
            # losing the real previous champion.
            raise ValueError(f"{version} is already the champion")
        if old:
            self.index["cards"][old]["status"] = "archived"
        self.index["previous_champion"] = old
        self.index["champion"] = version
        self.index["cards"][version]["status"] = "champion"
        self._event(when, version, "promoted", reason)
        self._save()

    def rollback(self, when: str, reason: str) -> str:
        bad, prev = self.index["champion"], self.index["previous_champion"]
        if not prev:
            raise RuntimeError("no previous champion to roll back to")
        self.index["cards"][bad]["status"] = "rolled_back"
        self.index["cards"][prev]["status"] = "champion"
        self.index["champion"], self.index["previous_champion"] = prev, None
        self._event(when, bad, "rolled_back", reason)
        self._save()
        return prev

    def update_live(self, version: str, metrics: dict) -> None:
        self.index["cards"][version]["live"] = metrics

    # -- reads ------------------------------------------------------------------------------
    def champion(self) -> str | None:
        return self.index["champion"]

    def card(self, version: str) -> dict:
        return self.index["cards"][version]

    def load(self, version: str):
        if version not in self._models:
            self._models[version] = joblib.load(self.root / version / "model.joblib")
        return self._models[version]

    def events(self) -> list[dict]:
        return list(self.index["events"])

    def next_version(self) -> str:
        return f"v{len(self.index['cards']) + 1:03d}"

    # -- internals --------------------------------------------------------------------------
    def _event(self, when: str, version: str, what: str, reason: str) -> None:
        self.index["events"].append({"when": when, "version": version, "event": what, "reason": reason})

    def _save(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=1, default=str))
