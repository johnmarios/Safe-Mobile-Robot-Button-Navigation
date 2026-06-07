from render.random_policy_render import run_random_policy
# from render.render_sac import render_sac

RENDERERS = {
    "random": run_random_policy,
    # "sac": render_sac,
}


def main():
    env_name = "random"
    # env_name = "sac"

    if env_name not in RENDERERS:
        raise ValueError(f"Unknown renderer: {env_name}")

    RENDERERS[env_name]()


if __name__ == "__main__":
    main()