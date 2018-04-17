#!/usr/bin/env python
import rospy
import itertools
import numpy as np
import openravepy as orpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# Local modules
import raveutils as ru


def plan_cartesian_twist(robot, twist, num_waypoints=10):
  """
  Plan the cartesian trajectory to apply the given twist to the current robot
  configuration

  Parameters
  ----------
  robot: orpy.Robot
    The OpenRAVE robot
  twist: array_like
    The twist to be applied to the end-effector
  num_waypoints: int
    Number of waypoints to be used by the trajectory

  Returns
  -------
  traj: orpy.Trajectory
    Planned trajectory. If plan fails, this function returns `None`.
  """
  waypoints = [robot.GetActiveDOFValues()]
  delta = np.array(twist) / float(num_waypoints)
  with robot:
    for i in xrange(num_waypoints):
      q = robot.GetActiveDOFValues()
      J = ru.kinematics.compute_jacobian(robot)
      Jpinv = np.linalg.pinv(J)
      delta_q = np.dot(Jpinv, delta)
      q += delta_q
      robot.SetActiveDOFValues(q)
      if ru.kinematics.check_joint_limits(robot, q):
        waypoints.append(q)
      else:
        # q is outside the joint limits
        return None
  traj = trajectory_from_waypoints(robot, waypoints)
  status = retime_trajectory(robot, traj, 'ParabolicTrajectoryRetimer')
  if status != orpy.PlannerStatus.HasSolution:
    # Time parameterization failed. Could not find a valid path solution.
    return None
  return traj

def plan_constant_velocity_twist(robot, twist, velocity, num_waypoints=10):
  """
  Plan the cartesian trajectory to apply the given twist to the current robot
  configuration at constant velocity

  Parameters
  ----------
  robot: orpy.Robot
    The OpenRAVE robot
  twist: array_like
    The twist to be applied to the end-effector
  velocity: float
    Robot velocity in the cartesian workspace (units are meters)
  num_waypoints: int
    Number of waypoints to be used by the trajectory

  Returns
  -------
  traj: orpy.Trajectory
    Planned trajectory. If plan fails, this function returns `None`.
  """
  manip = robot.GetActiveManipulator()
  indices = manip.GetArmIndices()
  DOF = robot.GetActiveDOF()
  velocity_limits = robot.GetDOFVelocityLimits()[indices]
  spec = orpy.ConfigurationSpecification()
  suffix = robot.GetName() + ' ' + ' '.join(map(str, indices))
  values_offset = spec.AddGroup('joint_values '+suffix, DOF, 'linear')
  velocities_offset = spec.AddGroup('joint_velocities '+suffix, DOF,'next')
  deltatime_offset = spec.AddGroup('deltatime', 1, '')
  traj = orpy.RaveCreateTrajectory(robot.GetEnv(), '')
  traj.Init(spec)
  # Add the current robot joint values
  waypoint = np.zeros(deltatime_offset+1)
  waypoint[values_offset:values_offset+DOF] = robot.GetActiveDOFValues()
  traj.Insert(0, waypoint)
  # Compute and populate the traj waypoints
  distance = np.linalg.norm(twist)
  duration = distance / velocity
  dt = duration / float(num_waypoints)
  delta = np.array(twist) / float(num_waypoints)
  with robot:
    for i in xrange(num_waypoints):
      q = robot.GetActiveDOFValues()
      J = ru.kinematics.compute_jacobian(robot)
      Jpinv = np.linalg.pinv(J)
      delta_q = np.dot(Jpinv, delta)
      q += delta_q
      qdot = delta_q / dt
      if not ru.kinematics.check_joint_limits(robot, q):
        break
      if np.any(np.abs(qdot) > velocity_limits):
        break
      robot.SetActiveDOFValues(q)
      # Populate the waypoint
      waypoint = np.zeros(deltatime_offset+1)
      waypoint[values_offset:values_offset+DOF] = q
      waypoint[velocities_offset:velocities_offset+DOF] = qdot*np.ones(DOF)
      waypoint[deltatime_offset] = dt
      traj.Insert(traj.GetNumWaypoints(), waypoint)
  if traj.GetNumWaypoints() < (num_waypoints-1):
    return None
  else:
    return traj

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
    If set, will compute the direct and reversed trajectory. The minimum
    duration trajectory is used.

  Returns
  -------
  traj: orpy.Trajectory
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
  reversed_is_better = False
  count = 0
  for qa, qb in itertools.permutations([qstart, qgoal], 2):
    count += 1
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
            if count == 2:
              reversed_is_better = True
    if not try_swap:
      break
  # Check if we need to reverse the trajectory
  if reversed_is_better:
    best_traj = orpy.planningutils.ReverseTrajectory(best_traj)
  return best_traj

def retime_trajectory(robot, traj, method):
  """
  Retime an OpenRAVE trajectory using the specified method.

  Parameters
  ----------
  robot: orpy.Robot
    The OpenRAVE robot
  traj: orpy.Trajectory
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
  status = orpy.PlannerStatus.Failed
  success = planner.InitPlan(robot, params)
  if success:
    status = planner.PlanPath(traj)
  return status

def ros_trajectory_from_openrave(robot_name, traj):
  """
  Convert an OpenRAVE trajectory into a ROS JointTrajectory message.

  Parameters
  ----------
  robot_name: str
    The robot name in OpenRAVE
  traj: orpy.Trajectory
    The input OpenRAVE trajectory

  Returns
  -------
  ros_traj: trajectory_msgs/JointTrajectory
    The equivalent ROS JointTrajectory message
  """
  ros_traj = JointTrajectory()
  # Specification groups
  spec = traj.GetConfigurationSpecification()
  try:
    values_group = spec.GetGroupFromName('joint_values {0}'.format(robot_name))
  except orpy.openrave_exception:
    orpy.RaveLogError('Corrupted traj, failed to find group: joint_values')
    return None
  try:
    velocities_group = spec.GetGroupFromName(
                                      'joint_velocities {0}'.format(robot_name))
  except orpy.openrave_exception:
    orpy.RaveLogError('Corrupted traj, failed to find group: joint_velocities')
    return None
  try:
    deltatime_group = spec.GetGroupFromName('deltatime')
  except orpy.openrave_exception:
    orpy.RaveLogError('Corrupted trajectory. Failed to find group: deltatime')
    return None
  # Copy waypoints
  time_from_start = 0
  skipped_time = 0
  num_waypoints = traj.GetNumWaypoints()
  for i in xrange(num_waypoints):
    waypoint = traj.GetWaypoint(i).tolist()
    deltatime = waypoint[deltatime_group.offset]
    time_from_start += deltatime
    if (i > 0) and np.isclose(deltatime, 0) and np.isclose(skipped_time, 0):
      skipped_time += deltatime
      continue
    else:
      skipped_time = 0
    # Append waypoint
    ros_point = JointTrajectoryPoint()
    values_end = values_group.offset + values_group.dof
    ros_point.positions = waypoint[values_group.offset:values_end]
    velocities_end = velocities_group.offset + velocities_group.dof
    ros_point.velocities = waypoint[velocities_group.offset:velocities_end]
    ros_point.time_from_start = rospy.Duration(time_from_start)
    ros_traj.points.append(ros_point)
  return ros_traj

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
  traj: orpy.Trajectory
    Resulting OpenRAVE trajectory.
  """
  env = robot.GetEnv()
  traj = orpy.RaveCreateTrajectory(env, '')
  traj.Init(robot.GetActiveConfigurationSpecification())
  for i,q in enumerate(waypoints):
    traj.Insert(i, q)
  return traj
