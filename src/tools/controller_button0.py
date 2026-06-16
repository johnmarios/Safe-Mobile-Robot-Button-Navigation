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

    def __init__(
        self,
        num_rays=16,
        max_speed=12.0,
        min_speed=2.0,
        max_steer=0.785,
        steer_gain=1.0,
        front_threshold=0.85,
        reverse_speed=-2.0,
        reverse_steps=30,
        reverse_cooldown_steps=40,
    ):
        self.num_rays = num_rays
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steer = max_steer
        self.steer_gain = steer_gain
        self.front_threshold = front_threshold
        self.reverse_speed = reverse_speed
        self.reverse_steps = reverse_steps
        self.reverse_cooldown_steps = reverse_cooldown_steps

        self.reverse_counter = 0
        self.reverse_cooldown = 0
        self.stored_reverse_steer = 0.0

        raw_angles = np.arange(num_rays) * 2 * np.pi / num_rays
        # [0, 2pi/16, 4pi/16, ..., 30pi/16, 32pi/16]


        # Normalize angles to be in the range [-pi, pi]
        self.angles = (raw_angles + np.pi) % (2 * np.pi) - np.pi 

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
    
    def compute_reverse_action(self, wrong_buttons_lidar, goal_angle):
        wrong_front_left = wrong_buttons_lidar[1]
        wrong_front_center = wrong_buttons_lidar[0]
        wrong_front_right = wrong_buttons_lidar[15]

        front_wrong = max(
            wrong_front_left,
            wrong_front_center,
            wrong_front_right,
        )
        # wrongside > 0 : the obstacle is on the left | < 0 : the obstacle is on the right
        wrong_side = wrong_front_left - wrong_front_right

        if abs(wrong_side) > 0.05: # in case the obstacle is on the left or right side, steer away from it
            reverse_steer = - np.sign(wrong_side) * self.max_steer
        else: # in case that the wrong button is straight ahead, steer away from the goal
            reverse_steer = np.clip(-0.7 * goal_angle,-self.max_steer,self.max_steer,)


        return reverse_steer, front_wrong

    def act(self, obs):
        buttons_lidar, goal_lidar = self.split_obs(obs)
        wrong_buttons_lidar = np.maximum(buttons_lidar - goal_lidar, 0.0)
        wrong_indices = np.where(wrong_buttons_lidar > 0.05)[0]
        
        goal_angle, goal_strength = self.get_lidar_to_angle(goal_lidar)

        reverse_steer, front_wrong = self.compute_reverse_action(
            wrong_buttons_lidar,
            goal_angle,
        )

        # 1. Αν ήδη κάνουμε όπισθεν, κάνε μόνο λίγα steps.
        if self.reverse_counter > 0:
            self.reverse_counter -= 1
            self.reverse_cooldown = self.reverse_cooldown_steps

            return np.array(
                [self.reverse_speed, self.stored_reverse_steer],
                dtype=np.float32
            )

        # 2. Μετά την όπισθεν, περίμενε λίγο πριν ξανακάνεις reverse.
        if self.reverse_cooldown > 0:
            self.reverse_cooldown -= 1

        # 3. Trigger reverse μόνο όταν το wrong button είναι αρκετά κοντά μπροστά.
        if front_wrong > self.front_threshold and self.reverse_cooldown == 0:
            self.reverse_counter = self.reverse_steps
            self.stored_reverse_steer = reverse_steer

            return np.array(
                [self.reverse_speed, self.stored_reverse_steer],
                dtype=np.float32
            )

        # 4. Normal mode: πήγαινε προς goal
        steer = self.steer_gain * goal_angle
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        turn_ratio = abs(steer) / self.max_steer
        front_factor = max(0.0, np.cos(goal_angle))

        # speed = self.max_speed * front_factor * (1.0 - 0.6 * turn_ratio)
        speed = self.max_speed 

        if goal_strength < 0.05:
            speed = self.min_speed

        # Αν υπάρχει wrong button μπροστά αλλά όχι αρκετά κοντά για reverse,
        # κόψε λίγο ταχύτητα.
        if front_wrong > 0.25:
            speed *= 0.85

        speed = np.clip(speed, self.min_speed, self.max_speed)

        return np.array([speed, steer], dtype=np.float32)