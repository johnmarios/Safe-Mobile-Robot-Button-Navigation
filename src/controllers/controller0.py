import numpy as np


class Button0Controller:
    """
    Reactive lidar controller for SafetyRacecarButton0-v0.

    Observation:
        obs[0:12]   -> racecar sensors
        obs[12:28]  -> buttons lidar
        obs[28:44]  -> goal lidar

    Lidar mapping:
        index 0  -> front
        index 4  -> left
        index 8  -> back
        index 12 -> right
        index 15 -> front-right

    Action:
        action[0] -> speed / wheel velocity
        action[1] -> steering

    the bigger the lidar value, the closer the object is.
    """


    MODE_RECOVERY = "recovery"
    MODE_REVERSE_TO_GOAL = "reverse_to_goal"
    MODE_AVOID = "avoid"
    MODE_SEEK_GOAL = "seek_goal"
    MODE_SEARCH = "search"

    def __init__(
        self,
        num_rays=16,
        max_speed=12.0,
        min_speed=2.0,
        max_steer=0.785,
        steer_gain=1.0,
        reverse_speed=-8.0,
        reverse_steps=30,
        reverse_cooldown_steps=10,
        avoid_threshold=0.35,
        recover_threshold=0.85,
        avoid_gain=0.5,
        search_steer=0.25,
        reverse_to_goal_threshold=2.0, 
        reverse_to_goal_speed=-10.0,
        reverse_to_goal_steer_gain=1.0,

        recovery_safe_threshold=0.85,
        recovery_min_steps=30,
        recovery_max_steps=1000,
        ):

        self.num_rays = num_rays
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steer = max_steer
        self.steer_gain = steer_gain
        # self.front_threshold = front_threshold
        self.reverse_speed = reverse_speed
        self.reverse_steps = reverse_steps
        self.reverse_cooldown_steps = reverse_cooldown_steps

        self.reverse_counter = 0
        self.reverse_cooldown = 0
        self.stored_reverse_steer = 0.0

        self.avoid_threshold = avoid_threshold
        self.recover_threshold = recover_threshold # when to reverse
        self.avoid_gain = avoid_gain
        self.search_steer = search_steer

        self.reverse_to_goal_threshold = reverse_to_goal_threshold
        self.reverse_to_goal_speed = reverse_to_goal_speed
        self.reverse_to_goal_steer_gain = reverse_to_goal_steer_gain

        # when it's safe to move forward again after reversing
        self.recovery_safe_threshold = recovery_safe_threshold

        self.recovery_min_steps = recovery_min_steps
        self.recovery_max_steps = recovery_max_steps

        self.recovery_active = False
        self.recovery_steps_done = 0

        self.last_goal_side = 1

        raw_angles = np.arange(num_rays) * 2 * np.pi / num_rays
        # [0, 2pi/16, 4pi/16, ..., 30pi/16, 32pi/16]


        # Normalize angles to be in the range [-pi, pi]
        self.angles = (raw_angles + np.pi) % (2 * np.pi) - np.pi 

    def wrap_angle(self, angle):
        # normalize angle to be in the range [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def split_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        # print("obs shape:", obs.shape)

        racecar_obs = obs[:12]
        # print("racecar_obs:", racecar_obs)

        task_obs = obs[12:]

        buttons_lidar = task_obs[:16]
        goal_lidar = task_obs[16:32]
        # print("buttons_lidar:", buttons_lidar)
        # print("goal_lidar:", goal_lidar)
        # print("goal_max:", np.max(goal_lidar))

        return buttons_lidar, goal_lidar
    
    def lidar_to_vector(self, lidar):
        x = np.sum(lidar * np.cos(self.angles))
        y = np.sum(lidar * np.sin(self.angles))
        return x, y

    def vector_to_angle(self, x, y):
        if abs(x) + abs(y) < 1e-6:
            return 0.0
        return np.arctan2(y, x)
    
    def get_lidar_to_angle(self, lidar):
        x, y = self.lidar_to_vector(lidar)
        goal_strength = np.max(lidar)
        goal_angle = self.vector_to_angle(x, y)
        return goal_angle, goal_strength
    
    def compute_features(self, buttons_lidar, goal_lidar):

        wrong_buttons_lidar = np.maximum(buttons_lidar - goal_lidar, 0.0)

        goal_angle, goal_strength = self.get_lidar_to_angle(goal_lidar)
        
        goal_side = self.get_goal_side(goal_angle, goal_strength)

        if goal_side != 0:
            self.last_goal_side = goal_side

        # far ahead for recovery
        front_danger = max(
            wrong_buttons_lidar[15],
            wrong_buttons_lidar[0],
            wrong_buttons_lidar[1],
        )

        # broader immediate front for avoidance
        forward_danger = max(
            wrong_buttons_lidar[14],
            wrong_buttons_lidar[15],
            wrong_buttons_lidar[0],
            wrong_buttons_lidar[1],
            wrong_buttons_lidar[2],
        )


        features = {
            "buttons_lidar": buttons_lidar,
            "goal_lidar": goal_lidar,
            "wrong_buttons_lidar": wrong_buttons_lidar,
            "goal_angle": goal_angle,
            "goal_strength": goal_strength,
            "goal_side": goal_side,
            "front_danger": front_danger,
            "forward_danger": forward_danger,
        }

        return features
    
    def select_mode(self, features):

        front_danger = features["front_danger"]

        # if we are in recovery mode, we continue until we have done enough steps or the front is safe enough
        if self.recovery_active:
            must_continue = self.recovery_steps_done < self.recovery_min_steps

            still_dangerous = (front_danger > self.recovery_safe_threshold and self.recovery_steps_done < self.recovery_max_steps)

            if must_continue or still_dangerous:
                return self.MODE_RECOVERY

            # recovery finished
            self.recovery_active = False
            self.recovery_steps_done = 0 # reset 
            self.reverse_cooldown = self.reverse_cooldown_steps

        # if we are in cooldown mode 
        if self.reverse_cooldown > 0:
            self.reverse_cooldown -= 1
            
        # if we are not in reverse mode, we check if we need to enter reverse mode
        if (front_danger > self.recover_threshold and self.reverse_cooldown == 0):
            self.recovery_active = True
            self.recovery_steps_done = 0
            self.stored_reverse_steer = self.compute_recovery_steer(features)
            return self.MODE_RECOVERY

        # if the goal is behind us, we go with rear mode towards it 
        if self.goal_is_behind(features):
            return self.MODE_REVERSE_TO_GOAL

        # if there is danger ahead, we avoid it 
        if features["forward_danger"] > self.avoid_threshold:
            return self.MODE_AVOID

        if features["goal_strength"] > 0.05:
            return self.MODE_SEEK_GOAL

        return self.MODE_SEARCH
    
    def compute_recovery_steer(self, features):
        goal_side = features["goal_side"]

        if goal_side == 0:
            goal_side = self.last_goal_side

        reverse_steer = -goal_side * self.max_steer

        return reverse_steer

    def compute_seek_goal_action(self, features):
        goal_angle = features["goal_angle"]
        goal_strength = features["goal_strength"]

        steer = self.steer_gain * goal_angle
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        speed = self.max_speed

        if goal_strength < 0.05:
            speed = self.min_speed

        return np.array([speed, steer], dtype=np.float32)

    
    def compute_recovery_action(self):
        # self.reverse_counter -= 1
        # self.reverse_cooldown = self.reverse_cooldown_steps
        self.recovery_steps_done += 1

        return np.array([self.reverse_speed, self.stored_reverse_steer],dtype=np.float32)
    
    def compute_avoid_action(self, features):
        """"
        we steer based on the goal and the wrong buttons,
        trying to avoid the wrong buttons while still heading towards the goal.

        """
        goal_lidar = features["goal_lidar"]
        wrong_buttons_lidar = features["wrong_buttons_lidar"]

        goal_x, goal_y = self.lidar_to_vector(goal_lidar)
        wrong_x, wrong_y = self.lidar_to_vector(wrong_buttons_lidar)

        desired_x = goal_x - self.avoid_gain * wrong_x
        desired_y = goal_y - self.avoid_gain * wrong_y
 
        desired_angle = self.vector_to_angle(desired_x, desired_y)

        steer = self.steer_gain * desired_angle
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        speed = 0.6 * self.max_speed
        speed = np.clip(speed, self.min_speed, self.max_speed)

        return np.array([speed, steer], dtype=np.float32)
    
    def compute_search_action(self):
        steer = self.last_goal_side * self.search_steer
        speed = self.min_speed

        return np.array([speed, steer], dtype=np.float32)
    
    def get_goal_side(self, goal_angle, goal_strength, angle_threshold=0.15):
        """
        Returns:
            1 -> goal is on the left
            -1 -> goal is on the right
            0 -> goal is almost centered or not visible enough
        """

        if goal_strength < 0.01:
            return 0

        if goal_angle > angle_threshold:
            return 1

        if goal_angle < -angle_threshold:
            return -1

        return 0
    
    def goal_is_behind(self, features):
        goal_angle = features["goal_angle"]
        goal_strength = features["goal_strength"]

        if goal_strength < 0.05:
            return False

        # if the goal is over 2.2 rad (approximately 126 degrees) away from the front, we consider it to be behind
        return abs(goal_angle) > self.reverse_to_goal_threshold
    
    def compute_reverse_to_goal_action(self, features):
        goal_angle = features["goal_angle"]

        # computes the error for steering when we are in rear mode to face the goal
        # 0 deg in front, 180 deg behind, 180 in front, 0 deg behind, 90 deg in front, -90 deg behind 
        if goal_angle >= 0:
            rear_error = self.wrap_angle(goal_angle - np.pi)
        else:
            rear_error = self.wrap_angle(goal_angle + np.pi)

        steer = -self.reverse_to_goal_steer_gain * rear_error
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        return np.array([self.reverse_to_goal_speed, steer],dtype=np.float32)

    def act(self, obs):
        buttons_lidar, goal_lidar = self.split_obs(obs)

        features = self.compute_features(buttons_lidar, goal_lidar)
        
        mode = self.select_mode(features)

        if mode == self.MODE_RECOVERY:
            return self.compute_recovery_action()
        
        if mode == self.MODE_REVERSE_TO_GOAL:
            return self.compute_reverse_to_goal_action(features)

        if mode == self.MODE_AVOID:
            return self.compute_avoid_action(features)

        if mode == self.MODE_SEEK_GOAL:
            return self.compute_seek_goal_action(features)

        if mode == self.MODE_SEARCH:
            return self.compute_search_action()

        return np.array([self.min_speed, 0.0], dtype=np.float32)