import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .filter import MonteCarloLocalizer
from .grid import GridMap
from .models import LidarModel, wrap
from .render import render
from .sim import Robot

DEFAULT_MAP = Path(__file__).parent / "maps" / "office.txt"


def first_sustained(mask, start=0, run=5):
    """First index >= start where mask stays true for `run` steps in a row."""
    streak = 0
    for i in range(start, len(mask)):
        streak = streak + 1 if mask[i] else 0
        if streak == run:
            return i - run + 1
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mcl",
                                 description="Global localization + kidnapped robot recovery with a particle filter.")
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--particles", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--kidnap-at", type=int, default=120, help="teleport the robot at this step (-1 to disable)")
    ap.add_argument("--beams", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--no-augment", action="store_true", help="plain MCL, no random particle injection")
    ap.add_argument("--live", action="store_true", help="redraw the map every step")
    ap.add_argument("--frames", default="0,3,10", help="steps to print a map for (non-live mode)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    grid = GridMap.load(args.map)
    lidar = LidarModel(n_beams=args.beams)
    rng = np.random.default_rng(args.seed)
    robot = Robot(grid, lidar, rng)
    pf = MonteCarloLocalizer(grid, args.particles, lidar=lidar, seed=args.seed + 1,
                             max_inject=0.0 if args.no_augment else 0.2)
    frames = {int(s) for s in args.frames.split(",") if s.strip()}
    if args.kidnap_at >= 0:
        frames |= {args.kidnap_at, args.kidnap_at + 3}

    print(f"map {grid.width_m:.1f} x {grid.height_m:.1f} m, {grid.w}x{grid.h} cells @ {grid.res:.2f} m, "
          f"{args.particles} particles, {args.beams} beams, augmented={'no' if args.no_augment else 'yes'}")

    errs, herrs = [], []
    t0 = time.perf_counter()
    for k in range(args.steps):
        if k == args.kidnap_at:
            robot.kidnap()
            if not args.quiet:
                print(f"\n>>> step {k}: robot kidnapped to ({robot.pose[0]:.1f}, {robot.pose[1]:.1f})")
        odom = robot.step()
        pf.predict(odom)
        pf.update(robot.scan())
        est = pf.estimate()

        err = float(np.hypot(est.x - robot.pose[0], est.y - robot.pose[1]))
        herr = float(abs(wrap(est.theta - robot.pose[2])))
        errs.append(err)
        herrs.append(herr)

        line = (f"step {k:3d}  err {err:6.2f} m  heading {np.degrees(herr):5.1f} deg  "
                f"conf {est.confidence:4.2f}  spread {pf.spread():5.2f} m  "
                f"n_eff {pf.n_eff():6.0f}  injected {pf.injected:4d}")
        if args.live:
            sys.stdout.write("\x1b[H\x1b[2J" + render(grid, pf.particles, robot.pose, est) + "\n" + line + "\n")
            sys.stdout.flush()
            time.sleep(0.05)
        elif not args.quiet:
            if k in frames:
                print("\n" + render(grid, pf.particles, robot.pose, est))
            if k in frames or k % 10 == 0:
                print(line)
    elapsed = time.perf_counter() - t0

    errs = np.array(errs)
    ok = errs < 0.3
    end = args.kidnap_at if args.kidnap_at >= 0 else len(errs)
    conv = first_sustained(ok[:end])
    print("\n--- summary ---")
    print(f"{args.steps} steps in {elapsed:.2f} s ({1000 * elapsed / args.steps:.1f} ms/step)")
    print(f"global localization: " + (f"converged at step {conv}" if conv is not None else "never converged"))
    if conv is not None:
        settled = errs[conv:end]
        print(f"tracking error after convergence: mean {settled.mean():.3f} m, max {settled.max():.3f} m, "
              f"mean heading {np.degrees(np.array(herrs)[conv:end]).mean():.2f} deg")
    if 0 <= args.kidnap_at < args.steps:
        rec = first_sustained(ok, start=args.kidnap_at)
        print("kidnap recovery: " + (f"relocalized after {rec - args.kidnap_at} steps" if rec is not None
                                     else "did not recover"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
