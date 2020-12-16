import gym


def get_hierarchical_task(args, logger):
    hierarchical_task = Hierarchical_Task(args, logger)
    return hierarchical_task


class Hierarchical_Task(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self._set_tasks()

    def __str__(self):
        for higher_task in self.tasks[1]:
            self.logger.log[self.args.log_name].info("Level 1: {}".format(higher_task))
        for lower_task in self.tasks[0]:
            self.logger.log[self.args.log_name].info("Level 0: {}".format(lower_task))
        return ""

    def _set_tasks(self):
        self.tasks = [[] for level in range(len(self.args.batch) - 1)]
        self._set_higher_tasks()
        self._set_lower_tasks()

    def _set_higher_tasks(self):
        if self.args.task == "empty":
            tasks = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-5x5-v0"]
        else:
            raise ValueError("Invalid task option")

        for task in tasks:
            self.tasks[1].append(gym.make(task))

    def _set_lower_tasks(self):
        for higher_task in self.tasks[1]:
            lower_tasks = higher_task.sample_tasks(num_tasks=self.args.batch[1])
            for lower_task in lower_tasks:
                self.tasks[0].append((higher_task, lower_task))

    def reset(self):
        self.tasks[0].clear()
        self._set_lower_tasks()
        print(self)

    def get_tasks(self):
        return self.tasks[0]

    def get_meta_tasks(self):
        return [
            self.tasks[0][0:int(len(self.tasks[0]) / 2.)],
            self.tasks[0][int(len(self.tasks[0]) / 2.):]]
