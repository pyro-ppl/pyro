from __future__ import absolute_import, division, print_function

import threading

import numpy as np

import math
import subprocess
import sys
import time
from collections import OrderedDict

try:
    from IPython.core.display import clear_output
    ipython_env = True
except (ModuleNotFoundError, ImportError):
    ipython_env = False

_write_lock = threading.Lock()
_instance_lock = threading.Lock()
_instance = None


class _SingleBar(object):
    def __init__(self,
                 total,
                 description=None,
                 postfix=None,
                 text_update=False,
                 dynamic_resize=False):
        self.total = total
        self.description = description
        self.postfix = postfix
        self._dynamic_resize = dynamic_resize
        self._text_update = text_update
        self._bar_width = 40
        self._elapsed_fill = "#"
        self._remaining_fill = "-"
        self._iter = 0
        self._elapsed_time = 0.0
        self._init_time = None
        self._update_interval = float(total) / self._bar_width
        self._term_width = self._get_term_width(refresh=True)

    @property
    def num_lines(self):
        screen_width = self._get_term_width()
        if screen_width in (None, 0):
            return 1
        return int(math.ceil(len(str(self)) / screen_width))

    @property
    def cls_str(self):
        return "\n".join([" " * self._get_term_width()] * self.num_lines)

    @property
    def bar(self):
        if self._iter == self.total:
            return self._elapsed_fill * self._bar_width
        else:
            elapsed = int(self._iter // self._update_interval)
            remaining = self._bar_width - elapsed
            return elapsed * self._elapsed_fill + remaining * self._remaining_fill

    def increment(self):
        self._iter += 1
        if self._init_time is None:
            self._init_time = time.time()
        else:
            self._elapsed_time = time.time() - self._init_time

    def _get_term_width(self, refresh=False):
        if not (refresh or self._dynamic_resize):
            return self._term_width
        cols = None
        try:
            cols = int(subprocess.check_output(["tput", "cols"], stderr=subprocess.STDOUT))
        except Exception:
            pass
        return cols

    def _gather_timing_stats(self):
        t = self._elapsed_time
        secs, mins, hrs, niters = 0, 0, 0, 0.0
        if t > 0:
            secs = int(t) % 60
            mins = int(t / 60) % 60
            hrs = int(t / 3600) % 24
            niters = self._iter / t
        return "{:d}:{:d}:{:d}".format(hrs, mins, secs), \
               "{:.2f} it/s".format(niters)

    def _format_postfix(self):
        postfix = self.postfix
        time_stats = self._gather_timing_stats()
        if postfix is None:
            return ", ".join(time_stats)
        postfix_stats = []
        for k, v in postfix.items():
            if isinstance(v, int):
                v = "{:d}".format(v)
            elif isinstance(v, float):
                v = "{:.3f}".format(v)
            elif not isinstance(v, str):
                raise ValueError("type must be in (int, str, float) for diagnostic values.")
            postfix_stats.append(k + "=" + v)
        return ", ".join(list(time_stats) + postfix_stats)

    def __str__(self):
        desc = "" if self.description is None else self.description + ": "
        if self._text_update:
            return "{}{:3.0f}% | {:d}/{:d} [ {} ]".format(desc,
                                                          float(self._iter / self.total) * 100,
                                                          self._iter,
                                                          self.total,
                                                          self._format_postfix())
        return "{}{:3.0f}% | {} | {:d}/{:d} [ {} ]".format(desc,
                                                           float(self._iter / self.total) * 100,
                                                           self.bar,
                                                           self._iter,
                                                           self.total,
                                                           self._format_postfix())


class ProgressBar(object):
    def __init__(self,
                 total,
                 description=None,
                 postfix=None,
                 dynamic_resize=False,
                 update_type="progress_bar",
                 num_bars=1):
        _instance_lock.acquire()
        global _instance
        assert _instance is None, "Only one instance of ProgressBar must be alive " \
                                  "at any given time."
        if isinstance(total, int):
            total = [total for _ in range(num_bars)]
        assert update_type in ("progress_bar", "text", None)
        text_update = update_type == "text"
        self.progress_bars = [_SingleBar(total[i], description, postfix, text_update, dynamic_resize)
                              for i in range(num_bars)]
        self.update_type = update_type
        self._iter = 0
        self._ref_pos, self._max_iters = np.argmax(total), max(total)
        self._text_update_interval = int(self._max_iters / 10)
        self._write("\n")
        if update_type == "progress_bar":
            self._write(str(self))
        _instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_num_lines(self):
        if self.update_type == "text":
            return None
        return sum(progress_bar.num_lines for progress_bar in self.progress_bars)

    def _get_cls_str(self):
        num_lines = self._get_num_lines()
        return self._init_point(num_lines) + \
            "\n".join([p.cls_str for p in self.progress_bars]) + \
            self._init_point(num_lines)

    def set_postfix(self, value, update=False, pos=0):
        assert isinstance(value, OrderedDict)
        self.progress_bars[pos].postfix = value
        if update:
            self.update()

    def set_description(self, value, update=False, pos=0):
        assert isinstance(value, str)
        self.progress_bars[pos].description = value
        if update:
            self.update()

    @classmethod
    def write(cls, msg, end="\n"):
        existing_progbbars = _instance and _instance.update_type == "progress_bar"
        with _write_lock:
            sys.stdout.flush()
            if ipython_env:
                try:
                    clear_output(wait=True)
                # for ipython console
                except Exception:
                    pass
            if existing_progbbars:
                sys.stdout.write(_instance._get_cls_str())
            sys.stdout.write(msg + end)
            if existing_progbbars:
                sys.stdout.flush()
                sys.stdout.write(str(_instance))

    def _write(self, msg):
        with _write_lock:
            sys.stdout.write(msg)

    @staticmethod
    def _init_point(num_lines):
        if num_lines == 0:
            return ""
        return "\033[F" * (num_lines - 1) if num_lines > 1 else "\r"

    def close(self):
        self._write("\n")
        global _instance
        _instance = None
        _instance_lock.release()

    def _text_update(self):
        if self._iter % self._text_update_interval == 0 or self._iter == self._max_iters:
            self._write(str(self) + "\n")

    def _bar_update(self):
        self._write(self._get_cls_str() + str(self))

    def update(self):
        if self.update_type is None:
            return
        elif self.update_type == "text":
            self._text_update()
        else:
            self._bar_update()

    def increment(self, pos=0, update=False):
        if self.update_type is None:
            return
        if pos == self._ref_pos:
            self._iter += 1
        self.progress_bars[pos].increment()
        if update:
            self.update()

    def __str__(self):
        return "\n".join([str(p) for p in self.progress_bars])


# from time import sleep
# p = ProgressBar(100, num_bars=2, update_type="progress_bar")
# for i in range(100):
#     sleep(0.01)
#     p.increment(pos=0)
#     p.increment(pos=1)
#     ProgressBar.write("bullshit")
#     p.update()
