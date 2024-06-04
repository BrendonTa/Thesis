import numpy as np

from Abb_robot.env.robot_env import MujocoRobotEnv
from Abb_robot.env.utils import rotations

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}
#purpose of this function is to check the full list of contacts that have been made by the geoms and see if the object and end effector has made contact
def block_grasped(self):
    grasp= False
    gripper_geom_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "robot0:suction_link")
    obj_geom_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, "object0")
    for i in range(self.data.ncon):
        contact = self.data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        if(geom1 == gripper_geom_id and geom2 == obj_geom_id) or (geom1 == obj_geom_id and geom2 == gripper_geom_id):
            grasp = True 
    return grasp
#will check the distance between two sites
def goal_distance(goal_a, goal_b): 
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
#Action logic behind the suction actuator
#This 
def check_position(grip_pos,object_pos,goal_pose,current_task):
    suction_on = False
    if (goal_distance(object_pos[:1],grip_pos[:1] )<0.1):
        if grip_pos[2]>object_pos[2]:
            suction_on = True
            
    elif goal_distance(goal_pose[:2],object_pos[:2])<0.3 and current_task == 1:
        suction_on = False
    return suction_on

def get_base_fetch_env(RobotEnvClass: MujocoRobotEnv):
    """Factory function that returns a BaseFetchEnv class that inherits
    from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings.
    """

    class BaseFetchEnv(RobotEnvClass):
        """Superclass for all Fetch environments."""

        def __init__(
            self,
            block_gripper,
            has_object: bool,
            distance_threshold,
            reward_type,
            **kwargs
        ):
            """Initializes a new Fetch environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
                has_object (boolean): whether or not the environment has an object
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            """
            self.block_gripper = block_gripper
            self.has_object = has_object
            self.distance_threshold = distance_threshold
            self.reward_type = reward_type

            super().__init__(n_actions=4, **kwargs)

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            # Compute distance between goal and the achieved goal.
            if self.current_task == 0: # if task is to pick goal is the object position achieved goal is the gripper position
                d = goal_distance(achieved_goal,goal)
                if block_grasped(self):
                    if self.reward_type == "sparse":
                        d +=1
                    else:
                        d+= 10
            # once picked up the goal is changed to the target site while the 
            elif self.current_task == 1:
                d = goal_distance(achieved_goal, goal)
                if d.all() < self.distance_threshold:
                    if self.reward_type == "sparse":
                        d +=1
                    else:
                        d+= 10

            if self.reward_type == "sparse": 
                return -(d > self.distance_threshold).astype(np.float64) 
            else:
                return -d.astype(np.float64) 

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            site_pos = np.array([1.2, 0.95, 0.3])
            assert action.shape == (4,)
            action = (
                action.copy()
            )  # ensure that we don't change the action outside of this scope
            pos_ctrl, gripper_ctrl = action[:3], action[3]

            pos_ctrl *= 0.05  # limit maximum change in position
            rot_ctrl = [
                1.0,
                0.0,
                1.0,
                0.0,
            ]  # fixed rotation of the end effector, expressed as a quaternion
            gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
            assert gripper_ctrl.shape == (2,)
            if self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            suction_state=check_position(grip_pos,object_pos,site_pos,self.current_task)
            if suction_state: 
                gripper_ctrl[1] = 1
                self.suction_on = suction_state

                
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

            return action

        def _get_obs(self):
            (
                grip_pos,
                object_pos,
                object_rel_pos,
                gripper_state,
                object_rot,
                object_velp,
                object_velr,
                grip_velp,
                gripper_vel,
                gripper_ctrl,
            ) = self.generate_mujoco_observations()

            if  self.current_task == 0:
                achieved_goal = grip_pos.copy()
            elif self.current_task == 1:
                achieved_goal = np.squeeze(object_pos.copy())

            obs = np.concatenate(
                [
                    grip_pos,
                    object_pos.ravel(),
                    object_rel_pos.ravel(),
                    gripper_state,
                    object_rot.ravel(),
                    object_velp.ravel(),
                    object_velr.ravel(),
                    grip_velp,
                    gripper_vel,
                    gripper_ctrl,
                ]
            )

            return {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(),
                "tasks": self.current_task
            }

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def _get_gripper_xpos(self):

            raise NotImplementedError

        def _sample_goal(self):
            if self.has_object:
                if self.current_task == 0: #pick task
                    goal = self._utils.get_site_xpos(self.model,self.data,"object0")
                elif self.current_task == 1: #place task
                    goal=np.array([1.2, 0.95, 0.3])
                     
            return goal.copy()

        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float64)

    return BaseFetchEnv

class MujocoFetchEnv(get_base_fetch_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        #ensures the suction link is fixed in the middle 
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:suction_link", 0.0
            )
        #change task if task if grasped
        if self.current_task == 0 and (block_grasped(self)) and self.suction_on:
            self.current_task = 1
            self.goal = self._sample_goal()
            self._render_callback()
            self.obs = self._get_obs()
        self._mujoco.mj_forward(self.model, self.data)



    def _set_action(self, action):
        action = super()._set_action(action)
        actuator_name ="robot0:adhere_gripper"
        actuator_id=self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id != -1:
            ctrl_value = action[8]
            self.data.ctrl[actuator_id] = ctrl_value
            self._mujoco.mj_forward(self.model, self.data)
        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)
       
        

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        #actuator state
        gripper_ctrl=[]
        actuator_id= self._mujoco.mj_name2id(self.model,self._mujoco.mjtObj.mjOBJ_ACTUATOR,"robot0:adhere_grip")
        gripper_ctrl.append(self.data.ctrl[actuator_id])
        
    

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
            gripper_ctrl,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:grip"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            #randomizes the state of the object position at the start of any simulation
            object_qpos[:2] += self.np_random.uniform(-0.08, 0.08, size=2)
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )
        self.current_task=0
        
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)
        # Move end effector into position.
        gripper_target = np.array(
            [0, 0, 0.1]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
       