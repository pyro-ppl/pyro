import sys
from time import sleep

class ProgressBar(object):
    def __init__(self, total, desc, postfix):
        self._width = 40
        self._fill = "="
        self._t = 0
        self._update_interval = total // self._width
        self.total = total
        self.desc = desc
        self.postfix = postfix
        self.progbar = ""

    def update(self):
        sys.stdout.flush()
        sys.stdout.write("\r" + str(self))

    def _format(self):
        postfix = str(self.postfix)
        return self.desc + " [" + self.progbar.ljust(self._width) + "] " + "\n({})".format(postfix)

    def increment(self):
        self._t += 1
        if self._t % self._update_interval == 0:
            self.progbar += self._fill
            self.update()

    def __str__(self):
        return self._format()

    
p = ProgressBar(total=100, desc="hello", postfix="bye")

for i in range(100):
    sleep(0.2)
    p.increment()
