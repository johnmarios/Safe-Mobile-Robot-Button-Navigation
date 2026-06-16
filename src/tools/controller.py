import os
import sys
import time
import argparse

import numpy as np
import safety_gymnasium


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_dir = os.path.join(src_dir, "train")

if src_dir not in sys.path:
    sys.path.append(src_dir)

if train_dir not in sys.path:
    sys.path.append(train_dir)

from train.normalize import NormalizeActionWrapper


def get_lidar_slices(obs):
    """
    Από το inspect:
    accelerometer: 3
    velocimeter:   3
    gyro:          3
    magnetometer:  3
    buttons_lidar: 16
    goal_lidar:    16

    Total = 44
    """
    obs = np.asarray(obs, dtype=np.float32)

    buttons_lidar = obs[12:28]
    goal_lidar = obs[28:44]

    velocimeter = obs[3:6]

    return buttons_lidar, goal_lidar, velocimeter


class LidarRacecarController:
    def __init__(
        self,
        target="goal",
        invert_steering=False,
        base_speed=0.45,
        fast_speed=0.85,
        slow_speed=0.20,
        steering_gain=1.5,
        reverse_steps=35,
        stuck_velocity_threshold=0.03,
        stuck_counter_limit=25,
    ):
        self.target = target
        self.invert_steering = invert_steering

        self.base_speed = base_speed
        self.fast_speed = fast_speed
        self.slow_speed = slow_speed
        self.steering_gain = steering_gain

        self.reverse_steps = reverse_steps
        self.reverse_counter = 0

        self.stuck_velocity_threshold = stuck_velocity_threshold
        self.stuck_counter_limit = stuck_counter_limit
        self.stuck_counter = 0

        self.last_steering = 0.0

    def choose_lidar(self, obs):
        buttons_lidar, goal_lidar, velocimeter = get_lidar_slices(obs)

        if self.target == "goal":
            lidar = goal_lidar
        elif self.target == "button":
            lidar = buttons_lidar
        else:
            raise ValueError("target must be 'goal' or 'button'")

        return lidar, velocimeter

    def compute_action(self, obs):
        lidar, velocimeter = self.choose_lidar(obs)

        velocity_norm = float(np.linalg.norm(velocimeter))

        # Αν έχουμε κολλήσει και δεν κινούμαστε, κάνε όπισθεν για λίγο.
        if velocity_norm < self.stuck_velocity_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter >= self.stuck_counter_limit and self.reverse_counter <= 0:
            self.reverse_counter = self.reverse_steps
            self.stuck_counter = 0
            print("[LIDAR CTRL] Stuck detected -> reversing")

        if self.reverse_counter > 0:
            self.reverse_counter -= 1

            speed = -1.0
            steering = -self.last_steering

            if abs(steering) < 0.3:
                steering = 0.7

            action = np.array([speed, steering], dtype=np.float32)
            return np.clip(action, -1.0, 1.0), lidar, velocity_norm, True

        max_value = float(np.max(lidar))
        max_index = int(np.argmax(lidar))

        num_bins = len(lidar)

        # bin_center: -1 αριστερά, 0 μπροστά, +1 δεξιά περίπου
        # Αν δεις ότι στρίβει ανάποδα, τρέξε με --invert-steering.
        center = (num_bins - 1) / 2.0
        bin_offset = (max_index - center) / center

        if self.invert_steering:
            bin_offset = -bin_offset

        steering = self.steering_gain * bin_offset
        steering = float(np.clip(steering, -1.0, 1.0))

        # Αν ο στόχος είναι κοντά στο κέντρο, πάμε πιο γρήγορα.
        abs_offset = abs(bin_offset)

        if max_value <= 1e-6:
            # Δεν βλέπει στόχο: προχώρα αργά και σκάναρε
            speed = self.slow_speed
            steering = 0.6
        elif abs_offset < 0.20:
            speed = self.fast_speed
        elif abs_offset < 0.55:
            speed = self.base_speed
        else:
            speed = self.slow_speed

        self.last_steering = steering

        action = np.array([speed, steering], dtype=np.float32)

        return np.clip(action, -1.0, 1.0), lidar, velocity_norm, False


def run_controller():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="SafetyRacecarButton0-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.01)

    parser.add_argument(
        "--target",
        type=str,
        default="goal",
        choices=["goal", "button"],
        help="Use goal_lidar or buttons_lidar.",
    )

    parser.add_argument(
        "--invert-steering",
        action="store_true",
        help="Use this if the car turns away from the target.",
    )

    args = parser.parse_args()

    env = safety_gymnasium.make(args.env_id, render_mode="human")
    env = NormalizeActionWrapper(env)

    print("\n==============================")
    print("Lidar Rule-based Racecar Controller")
    print("==============================")
    print(f"Env: {args.env_id}")
    print(f"Target lidar: {args.target}")
    print(f"Invert steering: {args.invert_steering}")
    print(f"Normalized action space: {env.action_space}")

    controller = LidarRacecarController(
        target=args.target,
        invert_steering=args.invert_steering,
    )

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)

        episode_reward = 0.0
        episode_cost = 0.0

        print(f"\nEpisode {episode + 1}/{args.episodes}")

        for step in range(args.max_steps):
            action, lidar, velocity_norm, reversing = controller.compute_action(obs)

            real_action = env.denormalize(action)

            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)
            episode_cost += float(cost)

            if step % 25 == 0:
                print(
                    f"step={step:4d} | "
                    f"lidar_max={np.max(lidar):.3f} | "
                    f"lidar_argmax={np.argmax(lidar):2d} | "
                    f"vel={velocity_norm:.3f} | "
                    f"norm_action={action} | "
                    f"real_action={real_action} | "
                    f"reverse={reversing} | "
                    f"reward={reward:.3f} | "
                    f"cost={cost}"
                )

            env.render()
            time.sleep(args.sleep)

            obs = next_obs

            done = terminated or truncated

            if done:
                print("Episode ended.")
                print(f"terminated={terminated}, truncated={truncated}")
                break

        print(f"Episode reward: {episode_reward:.3f}")
        print(f"Episode cost:   {episode_cost:.3f}")
        print(f"Episode steps:  {step + 1}")

    env.close()


if __name__ == "__main__":
    run_controller()