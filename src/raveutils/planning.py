#!/usr/bin/env python
import criros
import numpy as np
import openravepy as orpy


def plan_to_joint_configuration(robot, qgoal, planner='birrt', max_iters=20,
                                                                max_ppiters=40):
  """
  Plan the trajectory to the given ``qgoal`` configuration.

  Parameters
  ----------
  robot: orpy.Robot
    OpenRAVE robot object.
  qgoal: array_like
    The goal configuration.
  planner: str
    Name of the planning method.
  max_iters: float
    Maximum iterations of planning.
  max_ppiters: float
    Maximum iterations of post processing.

  Returns
  -------
  traj: orpy.trajectory
    Planned trajectory. If plan fails, return None.
  """
  env = robot.GetEnv()
  planner = orpy.RaveCreatePlanner(env, planner)
  params = orpy.Planner.PlannerParameters()
  params.SetRobotActiveJoints(robot)
  params.SetGoalConfig(qgoal)
  params.SetMaxIterations(max_iters)
  params.SetPostProcessing('ParabolicSmoother',
                '<_nmaxiterations>{0}</_nmaxiterations>'.format(max_ppiters))
  initsuccess = planner.InitPlan(robot, params)
  traj = None
  if initsuccess:
    # Plan a trajectory
    traj = orpy.RaveCreateTrajectory(env, '')
    status = planner.PlanPath(traj)
    if status != orpy.PlannerStatus.HasSolution:
      traj = None
  return traj
