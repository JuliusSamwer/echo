from dataclasses import dataclass


@dataclass
class MazeConfig:
    # geometry
    n_rooms_x: int = 3
    n_rooms_y: int = 3
    room_size: float = 1.0
    doorway_width: float = 0.18      # certified value (spec 0.22 was too permissive)
    doorway_jitter: float = 0.34     # OFF-CENTER doorways: the load-bearing fix. With centered
                                     # doorways a room-center subgoal is collinear with the gap, so
                                     # ADVERSARIAL threads it for free and placement never bites. See
                                     # the probe finding in config note below.
    # dynamics / action
    dynamics: str = "kinematic"      # "kinematic" | "double_integrator"
    max_step: float = 0.14           # s_max: max L2 magnitude of a primitive action
    dt: float = 0.1                  # used only for double_integrator
    agent_radius: float = 0.0        # >0 inflates walls for collision
    # success radii
    goal_eps: float = 0.08
    subgoal_eps: float = 0.08
    # geodesic field resolution (grid cells per room edge)
    geodesic_res: int = 60
    # rng
    seed: int = 0


# CERTIFIED REGIME (from the probe, 100 instances, L in {2,3,4}):
#   geometry: doorway_jitter=0.34, doorway_width=0.18
#   executor: H=10 budget, MPPIConfig(n_samples=384, n_iters=3, plan_horizon=14)
#   -> GT >> RANDOM, GT-ADVERSARIAL > 0 with 95% CIs excluding 0 at all L. PASS.
# The band is narrow: H=8 leaves GT unreliable; H>=11 collapses the GT-ADV gap (the
# low-level threads doorways unaided). Pick H=10.
#
# Why centered doorways fail: a doorway between two rooms sits on their shared
# center-line, so a room-center subgoal is collinear with the gap and the executor
# threads it regardless of precision -> placement goes free. doorway_jitter breaks
# that collinearity, which is what makes precise (doorway) placement causally matter.
