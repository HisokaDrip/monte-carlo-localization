# monte-carlo-localization

A particle filter that figures out where a robot is in an office floor plan using nothing but wheel odometry and a 30-beam lidar. It starts with no idea at all (3000 particles spread over the whole 25 x 11 m map), usually locks on within a dozen or so steps, and then I pick the robot up and drop it somewhere else to see if it notices. It does.

Everything is NumPy and plain Python, no ROS, no scipy:

- **Distance field** via the Felzenszwalb–Huttenlocher exact Euclidean distance transform (two 1-D lower-envelope passes). This is the likelihood field for the sensor model, and the simulator also uses it to sphere-trace the lidar rays instead of stepping one cell at a time.
- **Odometry motion model** (rot1 / trans / rot2 with noise that grows with how much you turned and drove).
- **Likelihood field sensor model** with a Gaussian hit term, a uniform junk term, and tempering, because 30 beams from the same scan are nowhere near independent and multiplying them raw makes the filter overconfident.
- **Low-variance (systematic) resampling**, triggered when the effective sample size drops below N/2.
- **Augmented MCL**: short- and long-term averages of how well the belief explains the scan. When the short-term one drops, a slice of particles gets re-seeded uniformly. That's what makes kidnap recovery work.
- **Cluster estimate**: the weighted mean is useless while the belief is multimodal (it lands inside the wall between two candidate rooms), so the estimate comes from the heaviest 1 m bin and its neighbours.
- **A simulated robot** that drives between random goals using A* on the distance field, with line-of-sight smoothing. Its real motion has slip the encoders don't report, and its lidar has noise plus random short returns.

## Running it

```
pip install -r requirements.txt
python -m mcl                 # default run: global localization, kidnap at step 120
python -m mcl --live          # redraws the map every step in your terminal
python -m mcl --no-augment    # plain MCL, for comparison
python -m mcl --help
pytest -q
```

In the map, `#` is a wall, `. : *` show particle density, `R` is where the robot really is, `E` is the filter's estimate, and `@` means they're in the same cell.

## A real run

`python -m mcl` (seed 9). This is trimmed, but the numbers are straight from the output.

Step 0: particles everywhere, and the estimate is just whichever bin happens to be heaviest.

```
##################################################
#..  .* *      #.     . .    ... # . ...*     .  #
# .           .#.               .#     ##     .  #
#     ###                        #.   .##.       #
#     ###                         ..             #
#     ...     .#       ####      #....    E.    .#
#  . . .  . . :#       ####     *###### .#########
#######..#######        :...    .#. . .       .:.#
#  ....    :.*.#*.      .**. : .:#:.    .       .#
#:.*:          ######## .#########.   .#*R       #
#* .##.       .  .. . :        .       #.:       #
#.. ##           .    .               .#####     #
# *.:*    .  ..#     ##..              .*.       #
#####:  ########     ##        ###     # .      :#
#           .:.#.    :       . ### .   #         #
#     .       .#  :.. :.     .  .:    *#    ##   #
#     #        #########   #########   #.   ##   #
#     #.            ..     . :.  ::    #         #
#     #       ..         .                       #
#              #     .. .##*.        .       .   #
#.    ..   .  :#  *.  .:.##::*      .*.#*.  :.   #
##################################################
step   0  err   2.18 m  heading  42.4 deg  conf 0.19  spread  7.81 m  n_eff   3000  injected    0
```

Step 3: most of the map has been ruled out, but it still prefers the wrong candidate down near the bottom wall.

```
##################################################
#      .       #                 #               #
#              #                 #     ##        #
#     ###                        #     ##        #
#     ###                                        #
#              #       ####      #         :     #
#              #       ####      ######  #########
#######  #######         .*      #               #
#   .          #                 #               #
#              ########  #########     # ..      #
#.  ##                                 #  R      #
#* .##                                 #####     #
#              #     ##                .         #
#####   ########    .##        ###     #         #
#              #               ###     #         #
#              #                       #    ##   #
#     #        #########   #########   #    ##   #
#     #                          .     #         #
#     #                                          #
#              #         ## E                    #
#              #   **    ##  ..        #         #
##################################################
step   3  err   8.46 m  heading   9.4 deg  conf 0.38  spread  6.35 m  n_eff   3000  injected    0
```

Step 10: locked on. The cloud has shrunk to a 3 cm spread around the right spot, and by step 20 the error is down to a few centimetres.

```
##################################################
#              #                 #               #
#              #                 #     ##        #
#     ###                        #     ##        #
#     ###                                        #
#              #       ####      #               #
#              #       ####      ######  #########
#######  #######                 #               #
#              #                 #               #
#              ########  #########     #         #
#   ##                                 #         #
#   ##                                 ##### E   #
#              #     ##                      R   #
#####   ########     ##        ###     #         #
#              #               ###     #         #
#              #                       #    ##   #
#     #        #########   #########   #    ##   #
#     #                                #         #
#     #                                          #
#              #         ##                      #
#              #         ##            #         #
##################################################
step  10  err   0.31 m  heading   7.6 deg  conf 1.00  spread  0.03 m  n_eff   3000  injected    0
step  20  err   0.07 m  heading   0.0 deg  conf 1.00  spread  0.08 m  n_eff   2013  injected    0
```

Then the kidnap. The cloud is still sitting where the robot used to be, the fit drops, and the filter starts re-seeding (`injected 565`). Eight steps later it has found the robot again.

```
>>> step 120: robot kidnapped to (9.5, 9.8)

##################################################
#    .    . .. #.  .  ..  . .    # .   ... .... .#
# ..   . . .   #.. . .     ....  #   . ##. ... . #
#.. ..###. . .  . .  ..  ...  . .#. .  ##. . ... #
#. .  ###  . .. ..  . . .  .  . ...      ...     #
#.  .. .  .....#. . ...#### . .. #... ...  .  ...#
# .  .....  .. # ..  . ####.. ...######  #########
#######..####### .. .      ... . # . .   .... .. #
# .. ... . .   #   .. . .  . ... #      .. . ..  #
# .   ....  . .######## .#########. .. #  ..     #
#  .##  ........ ...   . . ..  ...    .#  ..  . .#
#  .## . ........   .  .....     .. ...##### ..  #
# ..  .....  . # .  .## ... ....  .. ..   .  ... #
##### . ########     ##..  . ..###  . .#. ...   .#
#. ..  . ..    #...  .. ..  . .###. .  #.  ....  #
#    . . .. . .#.... . . .  .     .    #. ..##. .#
# ... # .. ..  #########...#########...# .  ##  .#
#.  . #   .. . .   . .. . .  .    ..  .# ..    . #
#. .  #   .E*..  .....   ..... ..  ..  ..  ... ..#
# . .. . .   ..#   R     ##  .. .. . . .. ...  ..#
# . .   ...  . #....... .##.   ..    ..# . .  .. #
##################################################
step 120  err   3.70 m  heading 116.3 deg  conf 0.82  spread  4.48 m  n_eff   3000  injected  565
...
step 130  err   0.17 m  heading   0.5 deg  conf 0.88  spread  2.81 m  n_eff   3000  injected  363
step 140  err   0.03 m  heading   0.6 deg  conf 1.00  spread  0.07 m  n_eff   2493  injected    0
step 150  err   0.02 m  heading   0.6 deg  conf 1.00  spread  0.08 m  n_eff   1598  injected    0
```

Summary:

```
--- summary ---
200 steps in 1.11 s (5.6 ms/step)
global localization: converged at step 11
tracking error after convergence: mean 0.044 m, max 0.279 m, mean heading 0.71 deg
kidnap recovery: relocalized after 8 steps
```

The same seed with `--no-augment`:

```
global localization: never converged
kidnap recovery: did not recover
```

What surprised me is that plain MCL doesn't even get through the *first* localization on this seed. It commits to the wrong room early on, the particles near the true pose get resampled away, and with no injection there's no way back. That's the particle deprivation problem the book warns about, and seeing it happen on a map this small made the augmented version click for me.

Across seeds 0–13 the augmented filter always converged (usually within 0–30 steps, worst case 47) and recovered from the kidnap in 1–28 steps. A full 200-step run takes about 1.1 s, or roughly 5–6 ms per step for 3000 particles × 30 beams.

## Layout

```
mcl/grid.py      occupancy grid, distance transform, sphere-traced raycast
mcl/models.py    motion + likelihood field models
mcl/filter.py    the localizer (augmented MCL, resampling, estimate)
mcl/planner.py   A* + shortcutting so the fake robot drives somewhere sensible
mcl/sim.py       ground-truth robot, noisy odometry, noisy lidar
mcl/render.py    ascii frames
mcl/maps/        maps are plain text, '#' = wall, 0.5 m per char
```

To try your own map, draw one in a text file and run `python -m mcl --map yourmap.txt`. Keep doorways at least two characters wide or the planner won't fit through them (I found that out the hard way).

## Notes / honest caveats

- The robot's controller steers on its true pose. A real robot would drive on the estimate, but then the simulation crashes into walls while the filter is still confused, and that's not what I was trying to test. The filter itself only ever sees odometry and scans.
- The map is the ground truth for both the simulator and the filter, so there's no map error. Real deployments always have some.
- The tempering factor (0.3) and the alpha_slow/alpha_fast values were tuned by eye on this map.

## What I'd do next

- KLD-sampling, so the particle count shrinks once it's confident (3000 is overkill while tracking).
- Feed the filter's estimate into the controller and see how much that hurts.
- A beam model with proper ray casting per particle, to compare against the likelihood field. It's more accurate near glass and clutter, but much slower.
- Load real occupancy maps (ROS `.pgm` + `.yaml`) instead of ASCII art.

MIT licensed.
