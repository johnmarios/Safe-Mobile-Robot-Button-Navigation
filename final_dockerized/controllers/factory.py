def create_controller(name, model_path=None, action_repeat=None):
    name = name.lower().replace("-", "_")

    if name == "random":
        from .random_controller import Controller
        return Controller()

    if name in {"rule", "rule_based"}:
        from .rule_based_controller import Controller
        return Controller()

    if name == "td3":
        from .td3_controller import Controller
        return Controller(model_path, action_repeat)

    if name == "sac":
        from .sac_controller import Controller
        return Controller(model_path, action_repeat)

    raise ValueError("Choose random, rule_based, td3 or sac.")
