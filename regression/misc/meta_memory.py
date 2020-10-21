class MetaMemory(object):
    def __init__(self):
        self.memories = {}

    def add(self, memory, key):
        self.memories[key] = memory

    def get(self, key):
        return self.memories[key]

    def clear(self):
        self.memories.clear()
