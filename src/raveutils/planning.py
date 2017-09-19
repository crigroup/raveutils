#!/usr/bin/env python
import itertools
import numpy as np
import openravepy as orpy


def plan_to_joint_configuration(robot, qgoal, pname='BiRRT', max_iters=20,
                                                max_ppiters=40, try_swap=False):
  """
  Plan a trajectory to the given `qgoal` configuration.

  Parameters
  ----------
  robot: orpy.Robot
    The OpenRAVE robot
  qgoal: array_like
    The goal configuration
  pname: str
    Name of the planning algorithm. Available options are: `BasicRRT`, `BiRRT`
  max_iters: float
    Maximum iterations for the planning stage
  max_ppiters: float
    Maximum iterations for the post-processing stage. It will use a parabolic
    smoother wich short-cuts the trajectory and then smooths it
  try_swap: bool
    If set will compute two trajectories: first `qstart` -> `qgoal`, second
    `qgoal` -> `qstart` and will select the minimum duration one.

  Returns
  -------
  traj: orpy.trajectory
    Planned trajectory. If plan fails, this function returns `None`.
  """
  qstart = robot.GetActiveDOFValues()
  env = robot.GetEnv()
  planner = orpy.RaveCreatePlanner(env, pname)
  params = orpy.Planner.PlannerParameters()
  params.SetMaxIterations(max_iters)
  if max_ppiters > 0:
    params.SetPostProcessing('ParabolicSmoother',
                '<_nmaxiterations>{0}</_nmaxiterations>'.format(max_ppiters))
  else:
    params.SetPostProcessing('', '')
  # Plan trajectory
  best_traj = None
  min_duration = float('inf')
  for qa, qb in itertools.permutations([qstart, qgoal], 2):
    with robot:
      robot.SetActiveDOFValues(qa)
      params.SetGoalConfig(qb)
      params.SetRobotActiveJoints(robot)
      initsuccess = planner.InitPlan(robot, params)
      if initsuccess:
        traj = orpy.RaveCreateTrajectory(env, '')
        status = planner.PlanPath(traj)             # Plan the trajectory
        if status == orpy.PlannerStatus.HasSolution:
          duration = traj.GetDuration()
          if duration < min_duration:
            min_duration = duration
            best_traj = orpy.RaveCreateTrajectory(env, traj.GetXMLId())
            best_traj.Clone(traj, 0)
    if not try_swap:
      break
  return best_traj

def retime_trajectory(robot, traj, method):
  """
  Retime an OpenRAVE trajectory using the specified method.

  Parameters
  ----------
  robot: orpy.Robot
    The OpenRAVE robot
  traj: orpy.trajectory
    The traj to be retimed. The time paremetrization will be *overwritten*.
  method: str
    Retiming method. Available options are: `LinearTrajectoryRetimer`,
    `ParabolicTrajectoryRetimer`, `CubicTrajectoryRetimer`

  Returns
  -------
  status: orpy.PlannerStatus
    Flag indicating the status of the trajectory retiming. It can be: `Failed`,
    `HasSolution`, `Interrupted` or `InterruptedWithSolution`.
  """
  env = robot.GetEnv()
  # Populate planner parameters
  params = orpy.Planner.PlannerParameters()
  params.SetRobotActiveJoints(robot)
  params.SetMaxIterations(20)
  params.SetPostProcessing('', '')
  # Generate the trajectory
  planner = orpy.RaveCreatePlanner(env, method)
  success = planner.InitPlan(robot, params)
  if success:
    status = planner.PlanPath(traj)
  else:
    status = orpy.PlannerStatus.Failed
  return status

def trajectory_from_waypoints(robot, waypoints):
  """
  Generate an OpenRAVE trajectory using the given waypoints.

  Parameters
  ----------
  robot: orpy.Robot
    The OpenRAVE robot
  waypoints: list
    List of waypoints (joint configurations)

  Returns
  -------
  traj: orpy.trajectory
    Resulting OpenRAVE trajectory.
  """
  env = robot.GetEnv()
  traj = orpy.RaveCreateTrajectory(env, '')
  traj.Init(robot.GetActiveConfigurationSpecification())
  for i,q in enumerate(waypoints):
    traj.Insert(i, q)
  return traj
