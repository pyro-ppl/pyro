from __future__ import absolute_import, division, print_function

import math
import subprocess
import sys

from time import sleep


class ProgressBar(object):
    def __init__(self, total, description=None, postfix=None, dynamic_resize=False, disable=False):
        self.total = total
        self.description = description
        self.postfix = postfix
        self.dynamic_resize = dynamic_resize
        self.disable = disable
        self._bar_width = 40
        self._fill = "="
        self._bar = ""
        self._t = 0
        self._update_interval = float(total) / self._bar_width
        self._screen_width = self._get_term_width()
        self._started = False

    @staticmethod
    def _get_term_width():
        cols = None
        try:
            cols = int(subprocess.check_output(["tput", "cols"]))
        except Exception:
            pass
        return cols

    @property
    def num_lines(self):
        screen_width = self._get_term_width()
        return int(math.ceil(len(str(self)) / screen_width)) if screen_width not in (None, 0) else None

    def update(self):
        if not self._started:
            sys.stdout.flush()
            sys.stdout.write("\n")
            self._started = True
        sys.stdout.write("\r" + str(self))

    def increment(self):
        self._t += 1
        if self._t == self.total:
            self._bar = self._fill * self.total
        else:
            self._bar = self._fill * int(self._t // self._update_interval)

    def _format_postfix(self):
        pass

    def __str__(self):
        postfix = str(self.postfix)
        return self.description + \
            " [" + self._bar.ljust(self._bar_width) + "] " + \
            "({})".format(self._format_postfix(postfix))


class MultiProgressBars(object):
    def __init__(self, progress_bars):
        self.progress_bars = progress_bars

    @property
    def num_lines(self):
        return sum(progress_bar.num_lines for progress_bar in self.progress_bars)

    def update(self):
        sys.stdout.flush()
        num_lines = self.num_lines
        if num_lines > 1:
            sys.stdout.write("\033[F" * (num_lines - 1))
        else:
            sys.stdout.write("\r")
        sys.stdout.write(str(self))

    def add(self, total, desc, postfix):
        self.progbars.append(ProgressBar(total, desc, postfix))

    def increment(self, pos=0):
        self.progbars[pos].increment()

    def __str__(self):
        return "\n".join([str(p) for p in self.progbars])


p = MultiProgressBars()
p.add(total=100, desc="1.", postfix="bye")
p.add(total=100, desc="2.", postfix="bye")
for i in range(100):
    sleep(0.2)
    p.increment(pos=0)
    p.increment(pos=1)
    p.update()
