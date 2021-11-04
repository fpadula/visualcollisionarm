import numpy as np
import torch
from kortex_api.TCPTransport import TCPTransport
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2
from kortex_api.Exceptions.KServerException import KServerException
import threading

# def parseConnectionArguments(parser = argparse.ArgumentParser()):
#     parser.add_argument("--ip", type=str, help="IP address of destination", default="192.168.1.10")
#     parser.add_argument("-u", "--username", type=str, help="username to login", default="admin")
#     parser.add_argument("-p", "--password", type=str, help="password to login", default="admin")
#     return parser.parse_args()

class DeviceConnection:

    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection(args):
        """
        returns RouterClient required to create services and send requests to device or sub-devices,
        """

        return DeviceConnection(args.ip, port=DeviceConnection.TCP_PORT, credentials=(args.username, args.password))

    @staticmethod
    def createUdpConnection(args):
        """
        returns RouterClient that allows to create services and send requests to a device or its sub-devices @ 1khz.
        """

        return DeviceConnection(args.ip, port=DeviceConnection.UDP_PORT, credentials=(args.username, args.password))

    def __init__(self, ipAddress, port=TCP_PORT, credentials = ("","")):

        self.ipAddress = ipAddress
        self.port = port
        self.credentials = credentials

        self.sessionManager = None

        # Setup API
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    # Called when entering 'with' statement
    def __enter__(self):

        self.transport.connect(self.ipAddress, self.port)

        if (self.credentials[0] != ""):
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000   # (milliseconds)
            session_info.connection_inactivity_timeout = 2000 # (milliseconds)

            self.sessionManager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ipAddress)
            self.sessionManager.CreateSession(session_info)

        return self.router

    # Called when exiting 'with' statement
    def __exit__(self, exc_type, exc_value, traceback):

        if self.sessionManager != None:

            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000

            self.sessionManager.CloseSession(router_options)

        self.transport.disconnect()

def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications
    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return

class KinovaEnv():
    def __init__(self, ipaddr, action_magnitude=1):
        # Maximum allowed waiting time during actions (in seconds)
        self.TIMEOUT_DURATION = 100
        self.router = DeviceConnection.createTcpConnection(ipaddr)
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)

        self.target_pos = np.array([0, 0, 0])
        self.obstacle_pos = np.array([0, 0, 0])
        self.agent_id = [0]
        
        self.max_linear_disp = 0.01 # In meters
        self.max_ang_disp = 2 # In degrees
        self.starting_angles = np.array([
            -106.9461,
            82.62139,
            0,
            -147.3734,
            35.54055,
            -30.09479,
            58.28131
        ])
        self.reset_arm_pose()

        j_angles = self.get_joint_angles()
        ee_pose = self.calc_fk(j_angles)        
        self.agent = {
            'pos': np.array([ee_pose.x, ee_pose.y, ee_pose.z]),
            'ori': np.array([ee_pose.theta_x, ee_pose.theta_y, ee_pose.theta_z])
        }

    def reset_arm_pose(self):
        return self.set_arm_angles(self.starting_angles)

    def get_joint_angles(self):
        try:            
            arm_joint_angles = self.base.GetMeasuredJointAngles()
        except KServerException as ex:
            print("Unable to get joint angles")
            print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
            print("Caught expected error: {}".format(ex))
            quit()   
        return arm_joint_angles

    def calc_fk(self, joint_angles):
        try:
            print("Computing Foward Kinematics using joint angles...")
            pose = self.base.ComputeForwardKinematics(joint_angles)
        except KServerException as ex:
            print("Unable to compute forward kinematics")
            print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
            print("Caught expected error: {}".format(ex))
            quit()  
        return pose

    def set_arm_angles(self, angles):
        robot_action = Base_pb2.Action()
        robot_action.name = "Agent movement (set angles)"
        robot_action.application_data = ""

        feedback = self.base_cyclic.RefreshFeedback()

        joint_angles = robot_action.reach_joint_angles.joint_angles
        for i in range(7):
            joint_angles[i].value = angles[i]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteAction(robot_action)
        finished = e.wait(self.TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        return finished

    def ori_as_quat(self, theta_x, theta_y, theta_z):
        return np.zeros((1, 4)) # Change latter

    def collect_observation(self):        
        visual_observation = np.zeros((64, 64)) # Change latter

        scalar_observation = []

        # Obstacle position
        scalar_observation.append(self.obstacle_pos[0])
        scalar_observation.append(self.obstacle_pos[1] - 1)
        scalar_observation.append(self.obstacle_pos[2])

        # Target position
        scalar_observation.append(self.target_pos[0])
        scalar_observation.append(self.target_pos[1] - 1)
        scalar_observation.append(self.target_pos[2])

        # Agent position
        scalar_observation.append(self.agent["pos"][0])
        scalar_observation.append(self.agent["pos"][1] - 1)
        scalar_observation.append(self.agent["pos"][2])

        # Agent orientation
        agent_ori_quat = self.ori_as_quat(self.agent["ori"])
        scalar_observation.append(agent_ori_quat[0])
        scalar_observation.append(agent_ori_quat[1])
        scalar_observation.append(agent_ori_quat[2])
        scalar_observation.append(agent_ori_quat[3])

        # Arm joint angles (needs to be between -1 and 1)
        arm_joint_angles = self.get_joint_angles()
        for i in range(7):
            if i != 2:
                scalar_observation.append(arm_joint_angles.joint_angles[i].value/180)        

        ee_pose = self.calc_fk(arm_joint_angles)
        # Arm end-effector position
        scalar_observation.append(ee_pose.x)
        scalar_observation.append(ee_pose.y - 1)
        scalar_observation.append(ee_pose.z)

        # Arm end-effector orientation
        ee_ori = self.ori_as_quat(ee_pose.theta_x, ee_pose.theta_y, ee_pose.theta_z)
        scalar_observation.append(ee_ori[0])
        scalar_observation.append(ee_ori[1])
        scalar_observation.append(ee_ori[2])
        scalar_observation.append(ee_ori[3])

        return [visual_observation, scalar_observation]

    def set_arm_pose(self, position, orientation):
        robot_action = Base_pb2.Action()
        robot_action.name = "Agent movement (set pose)"
        robot_action.application_data = ""

        feedback = self.base_cyclic.RefreshFeedback()

        cartesian_pose = robot_action.reach_pose.target_pose
        cartesian_pose.x = position[0]
        cartesian_pose.y = position[1]
        cartesian_pose.z = position[2]
        cartesian_pose.theta_x = orientation[0]
        cartesian_pose.theta_y = orientation[1]
        cartesian_pose.theta_z = orientation[2]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteAction(robot_action)
        finished = e.wait(self.TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        return finished

    def action_space_sample(self, size):
        actions = self.behavior_spec.create_random_action(size)
        # print("Creating random action for %d agents" % (size))
        # print(actions.shape)
        return actions

    def step(self, actions):
        # Sorting actions acording to the agent's ids
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        # action = delta_yaw, delta_pitch, delta_row, dx, dy, dz
        # Current robot position
        r_x, r_y, r_z = 0, 0, 0
        # Current robot rotation
        r_theta_x, r_theta_y, r_theta_z = 0, 0, 0
        # Target robot position
        tr_x, tr_y, tr_z = r_x + actions[3], r_y + actions[4], r_z + actions[5]
        # Target robot rotation
        tr_theta_x = r_theta_x + actions[1] 
        tr_theta_y = r_theta_y + actions[2]
        tr_theta_z = r_theta_z + actions[0]

        arm_completed_action = self.set_arm_pose([tr_x, tr_y, tr_z],
                                                [tr_theta_x, tr_theta_y, tr_theta_z])
        
        # If arm reached target set to True
        arm_on_target = False
        
        # Get state from camera
        state = None
        # Calculate reward
        reward = np.zeros((1,1))
        if arm_on_target:
            done = np.ones((1, 1))
            terminals = state, reward, done, self.agent_id
            decision = None
        else:
            done = np.zeros((1, 1))
            decision = state, reward, done, self.agent_id
            terminals = None

        return decision, terminals

    def reset(self):
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.behavior_name)
        states = decision_steps.obs
        for visual_i in self.visual_obs_indexes:
            states[visual_i] = states[visual_i].reshape((states[visual_i].shape[0],states[visual_i].shape[3],states[visual_i].shape[1],states[visual_i].shape[2]))
        agents_action = self.action_space_sample(self.no_of_agents)
        agents_prev_state = self.gen_empty_states()
        # step_prev_states, step_agents = env.reset()
        step_actions = np.array([])
        prev_state_was_terminal = np.repeat(False, self.no_of_agents)
        reset_variables = (agents_action, agents_prev_state, step_actions, prev_state_was_terminal, states, decision_steps.agent_id)
        return reset_variables
        # return states, decision_steps.agent_id


    def close(self):
        self._env.close()

    def gen_empty_states(self):
        ret = []
        for i in range(len(self.state_dim)):
            ret.append(np.zeros((self.no_of_agents,) + self.state_dim[i]))
        return ret

def main():
    kenv = KinovaEnv(ipaddr="192.168.0.10")
    obs = kenv.collect_observation()
    print(obs)

if __name__ == '__main__':
    main()