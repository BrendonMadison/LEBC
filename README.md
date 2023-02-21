# LEBC
# H1 Laser and Electron Beams: Compton (LEBC)

Computation and plotting of the differential cross-section and energy-angle spectrum in the electron rest frame and lab frame given a colliding laser beam and electron(positron) beam.

In the low center of mass energy regime this returns Thomson scattering. Though Thomson scattering is a "zero photon energy" approximation so you might not see an idealized case of this as it doesn't really exist!

Requires python3, numpy for computation, matplotlib for plotting.

Includes two new variable classes:

FourVec -- Handles Lorentz Four Vectors in Numpy in a fairly straightforward way

BeamVec -- Handles beam characteristics and random generation

And three functions if you want to be able to get the Compton scattering results in another program.

There are two files ... the standalone version is if you simply wish to import these functions or classes

The demo will display and generate plots for your viewing.
