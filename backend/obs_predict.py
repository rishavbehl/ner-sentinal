"""
Stage-1.5 of the pipeline: score the entire observation history with the
trained segment models and persist the results.

Why this exists — this is the most important engineering decision in the ML
stack, and it is worth saying out loud in the presentation:

The route-delay model consumes "risk on this route" as an input. If we train it
on the TRUE hazard (which we have, because we generated it) but serve it the
CLASSIFIER's predicted risk, the input distributions differ and the model is
silently wrong. We hit exactly that: a Siliguri→Gangtok route whose segments
summed to 3 h of delay was predicted at 27 h, because predicted failure
probability saturates near 1.0 where true hazard sat at 0.45.

The fix is to train the downstream model on the upstream model's OWN OUTPUTS —
a proper two-stage cascade. Stage 2 then sees at training time exactly the
distribution it will see in production.
"""
from __future__ import annotations

import os

import joblib
import numpy as np

from . import db
from .features import FEATURE_COLUMNS
from .inference import SEVERITY_RISKY

TABLE = """
CREATE TABLE IF NOT EXISTS obs_predictions (
    road_id TEXT, ts TEXT,
    pred_label INTEGER, pred_risk_score REAL, pred_severity REAL,
    pred_delay_hours REAL,
    PRIMARY KEY (road_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_obspred_ts ON obs_predictions(ts);
"""


def build(verbose: bool = True) -> int:
    with db.cursor(commit=True) as c:
        c.executescript(TABLE)
        c.execute("DELETE FROM obs_predictions")

    clf = joblib.load(os.path.join(db.ARTIFACT_DIR, "risk_clf.joblib"))
    reg = joblib.load(os.path.join(db.ARTIFACT_DIR, "delay_reg.joblib"))

    rows = db.query("SELECT road_id, ts," + ",".join(FEATURE_COLUMNS) +
                    " FROM observations")
    X = np.array([[r[c] for c in FEATURE_COLUMNS] for r in rows], dtype=np.float64)

    proba = clf.predict_proba(X)
    labels = np.argmax(proba, axis=1).astype(int)
    delays = np.maximum(reg.predict(X), 0.0)
    fail = np.clip(proba[:, 1] + proba[:, 2], 0.0, 1.0)
    sev = np.clip(SEVERITY_RISKY * proba[:, 1] + proba[:, 2], 0.0, 1.0)

    out = [(rows[i]["road_id"], rows[i]["ts"], int(labels[i]),
            round(float(fail[i]), 4), round(float(sev[i]), 4),
            round(float(delays[i]), 3)) for i in range(len(rows))]
    db.executemany(
        "INSERT OR REPLACE INTO obs_predictions VALUES (?,?,?,?,?,?)", out)
    if verbose:
        print(f"  obs_predictions: {len(out):,} rows "
              f"(mean severity {sev.mean():.3f}, mean delay {delays.mean():.2f} h)")
    return len(out)


if __name__ == "__main__":
    build()
