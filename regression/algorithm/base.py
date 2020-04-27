class Base(object):
    def __init__(self, args, log, tb_writer):
        super(Base, self).__init__()

        self.args = args
        self.log = log
        self.tb_writer = tb_writer
