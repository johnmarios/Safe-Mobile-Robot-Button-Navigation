import numpy as np


class Controller:
    """Reactive finite-state Button2 controller, adapted to reset()/act()."""

    MODE_RECOVERY = "RECOVERY"
    MODE_REVERSE_TO_GOAL = "REVERSE_TO_GOAL"
    MODE_AVOID = "AVOID"
    MODE_SEEK_GOAL = "SEEK_GOAL"
    MODE_SEARCH = "SEARCH"

    def __init__(
        self,
        num_rays: int = 16,
        max_speed: float = 12.0,
        min_speed: float = 2.0,
        max_steer: float = 0.785,
        steer_gain: float = 1.0,
        reverse_speed: float = -8.0,
        reverse_cooldown_steps: int = 10,
        avoid_threshold: float = 0.35,
        recover_threshold: float = 0.85,
        avoid_gain: float = 0.5,
        search_steer: float = 0.25,
        reverse_to_goal_threshold: float = 2.0,
        reverse_to_goal_speed: float = -10.0,
        reverse_to_goal_steer_gain: float = 1.0,
        recovery_safe_threshold: float = 0.85,
        recovery_min_steps: int = 30,
        recovery_max_steps: int = 1000,
    ) -> None:
        self.num_rays = int(num_rays)
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.max_steer = float(max_steer)
        self.steer_gain = float(steer_gain)
        self.reverse_speed = float(reverse_speed)
        self.reverse_cooldown_steps = int(reverse_cooldown_steps)
        self.avoid_threshold = float(avoid_threshold)
        self.recover_threshold = float(recover_threshold)
        self.avoid_gain = float(avoid_gain)
        self.search_steer = float(search_steer)
        self.reverse_to_goal_threshold = float(reverse_to_goal_threshold)
        self.reverse_to_goal_speed = float(reverse_to_goal_speed)
        self.reverse_to_goal_steer_gain = float(reverse_to_goal_steer_gain)
        self.recovery_safe_threshold = float(recovery_safe_threshold)
        self.recovery_min_steps = int(recovery_min_steps)
        self.recovery_max_steps = int(recovery_max_steps)

        raw_angles = np.arange(self.num_rays, dtype=np.float32) * (2.0 * np.pi / self.num_rays)
        self.angles = (raw_angles + np.pi) % (2.0 * np.pi) - np.pi
        self.reset()

    def reset(self, seed= None):
        self.reverse_cooldown = 0
        self.stored_reverse_steer = 0.0
        self.recovery_active = False
        self.recovery_steps_done = 0
        self.last_goal_side = 1

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _split_observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        # The original rule controller uses the first 12 car values, then 16 button
        # lidar values and 16 goal-lidar values
        buttons_lidar = observation[12:28]
        goal_lidar = observation[28:44]

        return buttons_lidar, goal_lidar
    def _lidar_to_vector(self, lidar: np.ndarray) -> tuple[float, float]:
        return float(np.sum(lidar * np.cos(self.angles))), float(np.sum(lidar * np.sin(self.angles)))

    @staticmethod
    def _vector_to_angle(x: float, y: float) -> float:
        if abs(x) + abs(y) < 1e-6:
            return 0.0
        return float(np.arctan2(y, x))

    def _goal_side(self, goal_angle: float, goal_strength: float, threshold: float = 0.15) -> int:
        if goal_strength < 0.01:
            return 0
        if goal_angle > threshold:
            return 1
        if goal_angle < -threshold:
            return -1
        return 0

    def _features(self, buttons_lidar, goal_lidar):
        wrong_buttons = np.maximum(buttons_lidar - goal_lidar, 0.0)
        goal_x, goal_y = self._lidar_to_vector(goal_lidar)
        goal_angle = self._vector_to_angle(goal_x, goal_y)
        goal_strength = float(np.max(goal_lidar))
        goal_side = self._goal_side(goal_angle, goal_strength)
        if goal_side != 0:
            self.last_goal_side = goal_side

        front_danger = float(max(wrong_buttons[15], wrong_buttons[0], wrong_buttons[1]))
        forward_danger = float(max(
            wrong_buttons[14], wrong_buttons[15], wrong_buttons[0], wrong_buttons[1], wrong_buttons[2]
        ))
        return {
            "buttons_lidar": buttons_lidar,
            "goal_lidar": goal_lidar,
            "wrong_buttons": wrong_buttons,
            "goal_angle": goal_angle,
            "goal_strength": goal_strength,
            "goal_side": goal_side,
            "front_danger": front_danger,
            "forward_danger": forward_danger,
        }

    def _select_mode(self, features):
        front_danger = float(features["front_danger"])
        if self.recovery_active:
            must_continue = self.recovery_steps_done < self.recovery_min_steps
            still_dangerous = (
                front_danger > self.recovery_safe_threshold
                and self.recovery_steps_done < self.recovery_max_steps
            )
            if must_continue or still_dangerous:
                return self.MODE_RECOVERY
            self.recovery_active = False
            self.recovery_steps_done = 0
            self.reverse_cooldown = self.reverse_cooldown_steps

        if self.reverse_cooldown > 0:
            self.reverse_cooldown -= 1

        if front_danger > self.recover_threshold and self.reverse_cooldown == 0:
            self.recovery_active = True
            self.recovery_steps_done = 0
            side = int(features["goal_side"]) or self.last_goal_side
            self.stored_reverse_steer = float(-side * self.max_steer)
            return self.MODE_RECOVERY

        if float(features["goal_strength"]) >= 0.05 and abs(float(features["goal_angle"])) > self.reverse_to_goal_threshold:
            return self.MODE_REVERSE_TO_GOAL
        if float(features["forward_danger"]) > self.avoid_threshold:
            return self.MODE_AVOID
        if float(features["goal_strength"]) > 0.05:
            return self.MODE_SEEK_GOAL
        return self.MODE_SEARCH

    def _seek_goal(self, features):
        steering = np.clip(self.steer_gain * float(features["goal_angle"]), -self.max_steer, self.max_steer)
        return np.array([self.max_speed, steering], dtype=np.float32)

    def _avoid(self, features):
        goal_x, goal_y = self._lidar_to_vector(features["goal_lidar"])
        wrong_x, wrong_y = self._lidar_to_vector(features["wrong_buttons"])
        desired_angle = self._vector_to_angle(goal_x - self.avoid_gain * wrong_x, goal_y - self.avoid_gain * wrong_y)
        steering = np.clip(self.steer_gain * desired_angle, -self.max_steer, self.max_steer)
        speed = np.clip(0.6 * self.max_speed, self.min_speed, self.max_speed)
        return np.array([speed, steering], dtype=np.float32)

    def _recovery(self) -> np.ndarray:
        self.recovery_steps_done += 1
        return np.array([self.reverse_speed, self.stored_reverse_steer], dtype=np.float32)

    def _reverse_to_goal(self, features):
        goal_angle = float(features["goal_angle"])
        rear_error = self._wrap_angle(goal_angle - np.pi) if goal_angle >= 0.0 else self._wrap_angle(goal_angle + np.pi)
        steering = np.clip(-self.reverse_to_goal_steer_gain * rear_error, -self.max_steer, self.max_steer)
        return np.array([self.reverse_to_goal_speed, steering], dtype=np.float32)

    def _search(self) -> np.ndarray:
        return np.array([self.min_speed, self.last_goal_side * self.search_steer], dtype=np.float32)

    def act(self, observation):
        buttons_lidar, goal_lidar = self._split_observation(observation)
        features = self._features(buttons_lidar, goal_lidar)
        mode = self._select_mode(features)

        if mode == self.MODE_RECOVERY:
            action = self._recovery()
        elif mode == self.MODE_REVERSE_TO_GOAL:
            action = self._reverse_to_goal(features)
        elif mode == self.MODE_AVOID:
            action = self._avoid(features)
        elif mode == self.MODE_SEEK_GOAL:
            action = self._seek_goal(features)
        else:
            action = self._search()

        info = {
            "controller": "rule_based",
            "mode": mode,
            "goal_angle": float(features["goal_angle"]),
            "goal_strength": float(features["goal_strength"]),
            "front_danger": float(features["front_danger"]),
            "forward_danger": float(features["forward_danger"]),
        }
        return action, info
