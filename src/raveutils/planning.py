#!/usr/bin/env python
import criros
import numpy as np
import openravepy as orpy


def plan_to_joint_configuration(robot, qgoal, pname='birrt', max_iters=20,
                                                                max_ppiters=40):
  """
  Plan the trajectory to the given ``qgoal`` configuration.

  Parameters
  ----------
  robot: orpy.Robot
    OpenRAVE robot object
  qgoal: array_like
    The goal configuration
  pname: str
    Name of the planning algorithm
  max_iters: float
    Maximum iterations of planning
  max_ppiters: float
    Maximum iterations of post-processing

  Returns
  -------
  traj: orpy.trajectory
    Planned trajectory. If plan fails, return None.
  """
  env = robot.GetEnv()
  planner = orpy.RaveCreatePlanner(env, pname)
  params = orpy.Planner.PlannerParameters()
  params.SetRobotActiveJoints(robot)
  params.SetGoalConfig(qgoal)
  params.SetMaxIterations(max_iters)
  if max_ppiters > 0:
    params.SetPostProcessing('ParabolicSmoother',
                '<_nmaxiterations>{0}</_nmaxiterations>'.format(max_ppiters))
  else:
    params.SetPostProcessing('', '')
  initsuccess = planner.InitPlan(robot, params)
  traj = None
  if initsuccess:
    # Plan a trajectory
    traj = orpy.RaveCreateTrajectory(env, '')
    status = planner.PlanPath(traj)
    if status != orpy.PlannerStatus.HasSolution:
      traj = None
  return traj

def retime_trajectory(robot, traj, method):
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
  env = robot.GetEnv()
  traj = orpy.RaveCreateTrajectory(env, '')
  traj.Init(robot.GetActiveConfigurationSpecification())
  for i in xrange(len(waypoints)):
    traj.Insert(i, waypoints[i])
  return traj
