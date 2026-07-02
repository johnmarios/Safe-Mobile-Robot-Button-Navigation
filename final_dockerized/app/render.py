import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import safety_gymnasium

from controllers.common import ENV_ID
from controllers.factory import create_controller


OUTPUT_DIR = Path("/app/outputs")
MODELS_DIR = Path("/app/models")


class VideoWriter:
    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.process = None

    def write(self, frame):
        frame = np.asarray(frame[..., :3], dtype=np.uint8)

        if self.process is None:
            height, width, _ = frame.shape
            command = [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(self.path),
            ]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

        self.process.stdin.write(frame.tobytes())

    def close(self):
        self.process.stdin.close()
        self.process.wait()


def make_env():
    return safety_gymnasium.make(
        ENV_ID,
        render_mode="rgb_array",
        width=960,
        height=960,
        camera_name="fixedfar",
    )


def render_episode(controller_name, controller, episode, seed, fps, render_every):
    env = make_env()
    video_path = OUTPUT_DIR / f"{controller_name}_seed{seed}_episode{episode}.mp4"
    writer = VideoWriter(video_path, fps)

    observation, _ = env.reset(seed=seed)
    controller.reset(seed=seed)
    writer.write(env.render())

    total_reward = 0.0
    total_cost = 0.0
    steps = 0
    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action, _ = controller.act(observation)
        observation, reward, cost, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_cost += cost
        steps += 1

        if steps % render_every == 0 or terminated or truncated:
            writer.write(env.render())

    writer.close()
    env.close()

    return {
        "controller": controller_name,
        "episode": episode,
        "seed": seed,
        "steps": steps,
        "reward": total_reward,
        "cost": total_cost,
        "goal_met": bool(info.get("goal_met", False)),
        "video": str(video_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", choices=["td3", "sac", "rule_based", "random"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--action-repeat", type=int)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / args.model if args.model else None
    controller = create_controller(
        args.controller,
        model_path,
        args.action_repeat,
    )

    results = []

    for episode in range(1, args.episodes + 1):
        result = render_episode(
            args.controller,
            controller,
            episode,
            args.seed + episode - 1,
            args.fps,
            args.render_every,
        )
        results.append(result)
        print(result)

    metrics_path = OUTPUT_DIR / f"{args.controller}_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
