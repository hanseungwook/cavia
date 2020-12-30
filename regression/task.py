import gym
import gym_env  # noqa


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
        tasks = ["2DNavigation-v0", "2DNavigationRot-v0"]
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
        meta_tasks = []
        for i_meta_task in range(len(self.tasks[1])):
            meta_tasks.append(
                self.tasks[0][i_meta_task * self.args.batch[1]:(i_meta_task + 1) * self.args.batch[1]])
        return meta_tasks
