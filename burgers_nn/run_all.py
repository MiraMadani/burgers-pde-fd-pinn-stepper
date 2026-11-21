"""Run everything: FD checks -> PINN -> learned stepper -> results/summary.md.
Use --quick for a fast smoke test (tiny training budgets; numbers are NOT meaningful)."""
import argparse

import report
import run_fd
import run_pinn
import run_stepper

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--pinn-iters", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run_fd.main()
    run_pinn.main(300 if a.quick else a.pinn_iters, a.seed)
    run_stepper.main(a.quick)
    report.main()
