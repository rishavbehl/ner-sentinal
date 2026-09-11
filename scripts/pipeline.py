#!/usr/bin/env python3
"""
One command to build the whole system from nothing.

    python3 scripts/pipeline.py

Order matters, and the order is the point:

  1. datagen.build_all()        real network + simulated weather/incidents/labels
  2. train.main()               STAGE 1: segment risk classifier + delay regressor
  3. obs_predict.build()        score the full history with the stage-1 models
  4. datagen.generate_route_samples()   route training set built on PREDICTIONS
  5. train.train_route_model()  STAGE 2: route-level delay regressor
  6. train.global_importance()  permutation importance on the held-out fold
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import datagen, obs_predict, train      # noqa: E402


def main(skip_data: bool = False) -> None:
    t0 = time.time()
    print("=" * 72)
    print("NER LOGISTICS SENTINEL — build pipeline")
    print("=" * 72)

    if not skip_data:
        print("\n[1/6] Building network + synthetic observation history ...")
        datagen.build_all()

    print("\n[2/6] STAGE 1 — training segment risk + delay models ...")
    train.main()

    print("\n[3/6] Scoring observation history with stage-1 models ...")
    obs_predict.build()

    print("\n[4/6] Building route-level training set from PREDICTED risk ...")
    datagen.generate_route_samples()

    print("\n[5/6] STAGE 2 — training route delay model ...")
    train.train_route_model()

    print("\n[6/6] Computing permutation importance on held-out fold ...")
    rep = train.global_importance()
    print("      top drivers:", ", ".join(
        d["feature"] for d in rep["global_importance_permutation"][:6]))

    print("\n" + "=" * 72)
    print(f"BUILD COMPLETE in {time.time() - t0:.0f}s")
    print("Start the server with:  ./run.sh       (or python3 -m backend.app)")
    print("=" * 72)


if __name__ == "__main__":
    main(skip_data="--skip-data" in sys.argv)
