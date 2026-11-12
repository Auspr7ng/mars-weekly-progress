# Cart model validation feedback — draft only

I downloaded `misc_3d_model/cart_v2.xml` and checked it in MuJoCo 3.8.1.

The original file does not compile because the four caster swivel bodies have moving joints but no mass/inertia of their own. The first error points to `caster_fl`. I kept the source unchanged and used a separate test copy with explicitly assumed fork inertials and a floor to investigate motion.

With that test copy, I did not observe numerical instability or tipping in the tested cases. However, the cart has strong rolling resistance and slow caster reorientation. With aligned wheels, a 10 N handle push produced about 0.024 m/s forward speed. With all casters initially at 90 degrees, the same push produced about 0.0031 m/s; the caster angles changed by only a few degrees. The roll-joint damping (0.05) and swivel damping/friction loss (0.8 / 0.2) should be checked against the intended hardware behavior.

Could you confirm the intended caster fork mass/inertia, bearing resistance, trail direction, and total cart mass? The temporary inertials used for testing are not hardware measurements.

This week's work was limited to independent model validation. Robot-cart interaction and the pushing task are planned for next week.
