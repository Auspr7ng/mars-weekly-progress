# Independent cart model validation

November 12, 2026. Scope: model download, structure audit and isolated physics checks. Robot-cart pushing is next week's work.

**Original source: compile failure.** The unchanged [cart_v2.xml](source/cart_v2.xml) has moving caster bodies without their own mass/inertia. MuJoCo 3.8.1 rejects the model.

A [diagnostic scene](results/cart_test_fork_0.05kg.xml) adds an assumed 0.05 kg fork mass per caster, stated inertias and a floor. These are test assumptions, not calibrated repairs. Eleven base cases and two mass-sensitivity cases remained finite without warnings or tipping. Motion shows strong resistance and slow caster reorientation.

- [Validation report](Validation_Report.md)
- [English feedback draft](Feedback_EN.md)
- [36-second video](results/cart_validation.mp4)
- [Numerical summary](results/validation_summary.json)
- [Motion comparison](results/motion_comparison.png)
- [Reproducible validation code](code/validate_cart.py)

```powershell
python -B code/validate_cart.py
python -B code/validate_cart.py --video
python -B code/plot_results.py
```

Run from this folder in the compatible MuJoCo environment. The script writes `results/`, preserves the source, and applies external test forces without a robot or learned task controller. The standalone source has no floor; the test harness supplies one.
