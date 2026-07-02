import os

from controllers.factory import create_controller


SELECTED_CONTROLLER = os.getenv("CONTROLLER_NAME", "td3")


class Controller:
    def __init__(self):
        self.controller = create_controller(SELECTED_CONTROLLER)

    def reset(self, seed=None):
        self.controller.reset(seed)

    def act(self, observation):
        return self.controller.act(observation)
