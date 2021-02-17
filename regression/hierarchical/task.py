import gym
import gym_env  # noqa


class HierarchicalTask(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self._set_meta_tasks()
        self._set_tasks()

    def __str__(self):
        for meta_task in self.meta_tasks:
            self.logger.log[self.args.log_name].info("Meta-Task: {}".format(meta_task))
        for task in self.tasks:
            self.logger.log[self.args.log_name].info("Task: {}".format(task))
        return ""

    def _set_meta_tasks(self):
        self.meta_tasks = [gym.make(meta_task) for meta_task in self.args.task]

    def _set_tasks(self):
        self.tasks = []
        for meta_task in self.meta_tasks:
            tasks = meta_task.sample_tasks(num_tasks=self.args.batch[1])
            for task in tasks:
                self.tasks.append((meta_task, task))

    def reset(self):
        self.tasks.clear()
        self._set_tasks()
        print(self)

    def get_tasks(self):
        return self.tasks

    def get_meta_tasks(self):
        meta_tasks = []
        for i_meta_task in range(len(self.meta_tasks)):
            meta_tasks.append(
                self.tasks[i_meta_task * self.args.batch[1]:(i_meta_task + 1) * self.args.batch[1]])
        return meta_tasks
