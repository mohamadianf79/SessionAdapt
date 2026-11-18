"""Shows which result files already exist and prints the exact command for each missing one.
Run from the project root:  python scripts/check_results.py"""
from pathlib import Path
R = Path("results")
# (file pattern, description, command template)  -- {c}=config path, {n}=config name
JOBS = {
 "yoochoose_f64": [
  ("{n}_baselines_test.csv", "Pop/S-Pop/Markov/SKNN test", "python scripts/run_baselines.py configs/{n}.yaml --final"),
  ("{n}_curve_test.csv", "cold-start curve (classical)", "python scripts/run_curve.py configs/{n}.yaml --final --models pop,spop,markov"),
  ("{n}_shift_diagnosis_val.csv", "shift diagnosis", "python scripts/diagnose_shift.py configs/{n}.yaml"),
  ("{n}_adaptive_test.csv", "score mixing + detector AUC", "python scripts/run_adaptive.py configs/{n}.yaml --final"),
  ("{n}_slots_test.csv", "slot reservation", "python scripts/run_slots.py configs/{n}.yaml --final"),
  ("{n}_backoff_test.csv", "category back-off", "python scripts/run_backoff.py configs/{n}.yaml --final"),
  ("{n}_detector_auc_val.csv", "detector AUC (written by run_adaptive)", "python scripts/run_adaptive.py configs/{n}.yaml"),
 ],
 "yoochoose_f8": [
  ("{n}_time_daypart4_wkend_test.csv", "time prior", "python scripts/run_time.py configs/{n}.yaml --final --scheme daypart4_wkend"),
 ],
 "diginetica": [
  ("{n}_baselines_test.csv", "Pop/S-Pop/Markov/SKNN test", "python scripts/run_baselines.py configs/{n}.yaml --final"),
  ("{n}_curve_test.csv", "cold-start curve (classical)", "python scripts/run_curve.py configs/{n}.yaml --final --models pop,spop,markov"),
  ("{n}_shift_diagnosis_val.csv", "shift diagnosis", "python scripts/diagnose_shift.py configs/{n}.yaml"),
  ("{n}_adaptive_test.csv", "score mixing + detector AUC", "python scripts/run_adaptive.py configs/{n}.yaml --final"),
  ("{n}_backoff_test.csv", "category back-off", "python scripts/run_backoff.py configs/{n}.yaml --final"),
  ("{n}_detector_auc_val.csv", "detector AUC", "python scripts/run_adaptive.py configs/{n}.yaml"),
  ("{n}_time_dow7_test.csv", "time prior (day of week)", "python scripts/run_time.py configs/{n}.yaml --final --scheme dow7"),
  ("{n}_neural_vs_markov_test.csv", "GRU/SASRec vs Markov (paired CIs)", "python scripts/compare_models.py configs/{n}.yaml"),
 ],
}
missing = 0
for n, jobs in JOBS.items():
    print(f"\n=== {n}")
    for pat, desc, cmd in jobs:
        f = R / pat.format(n=n); ok = f.exists()
        print(f"  [{'OK ' if ok else 'MISSING'}] {desc:42s} {f.name}")
        if not ok:
            missing += 1; print(f"            -> {cmd.format(n=n)}")
print(f"\n{missing} file(s) missing." + ("  Then run: python scripts/make_all_figures.py" if missing == 0 else ""))