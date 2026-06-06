from render.random_policy_render import run_random_policy

RENDERERS = {
    "random": run_random_policy,
}


def main():
    env_name = "random"

    if env_name not in RENDERERS:
        raise ValueError(f"Unknown renderer: {env_name}")

    RENDERERS[env_name]()


if __name__ == "__main__":
    main()