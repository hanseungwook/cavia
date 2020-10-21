import json


class StatusMonitor(object):
    def __init__(self, level_max):
        self.level_max = level_max

        self.status = {}
        self.status["iteration"] = [0 for _ in range(level_max)]
        self.status["i_task"] = [0 for _ in range(level_max)]

    def update(self, name, level, value):
        self.status[name][level] = value

    def key(self):
        return json.dumps(self.status)

    def clear(self):
        self.status["iteration"] = [0 for _ in range(self.level_max)]
        self.status["i_task"] = [0 for _ in range(self.level_max)]
