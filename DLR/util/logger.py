import os

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        if not os.path.exists(output_name):
            self.log_file = open(output_name, 'w')
        else:
            self.log_file = open(output_name, 'a')

        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def write(self, msg,p=False):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        if p:
            print(msg)
