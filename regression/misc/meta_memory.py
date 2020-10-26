class MetaMemory(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.memories = {}

    def _check_duplicate_key(self, key):
        for stored_key in self.memories.keys():
            if stored_key == key:
                return False
        return True

    def add(self, memory, key):
        if self._check_duplicate_key(key):
            self.memories[key] = memory
            self.logger.log[self.args.log_name].info(
                "[Meta-Memory] Added memory with key {}".format(key))
            self.logger.log[self.args.log_name].info(
                "[Meta-Memory] Memory size is {}".format(len(self.memories)))

    def get(self, key):
        return self.memories[key]

    def clear(self):
        self.memories.clear()
