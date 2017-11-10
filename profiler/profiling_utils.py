from __future__ import absolute_import, division, print_function

import cProfile
import functools
import os
import pstats
import timeit
from contextlib import contextmanager
from cStringIO import StringIO

from prettytable import ALL, PrettyTable

FILE = os.path.abspath(__file__)
PROF_DIR = os.path.join(os.path.dirname(FILE), 'data')
if not os.path.exists(PROF_DIR):
    os.makedirs(PROF_DIR)


class ProfilePrinter(object):

    def __init__(self, column_widths=None, field_format=None, template='column'):
        assert template in ('column', 'row')
        self._template = template
        self._column_widths = column_widths
        self._field_format = field_format
        self._header = None
        if template == 'column':
            self.table = PrettyTable(header=False, hrules=ALL)
        else:
            self.table = PrettyTable(header=False, hrules=ALL)

    def _formatted_values(self, values):
        if self._field_format is not None:
            assert len(self._field_format) == len(values)
            return [f.format(val) if f else str(val) for f, val in zip(self._field_format, values)]
        return values

    def _add_using_row_format(self, values):
        assert len(self._header) == len(values)
        formatted_vals = self._formatted_values(values)
        for i in range(len(self._header)):
            self.table.add_row([self._header[i], formatted_vals[i]])

    def _add_using_column_format(self, values):
        formatted_vals = self._formatted_values(values)
        self.table.add_row(formatted_vals)

    def push(self, values):
        if self._template == 'column':
            self._add_using_column_format(values)
        else:
            self._add_using_row_format(values)

    def header(self, values):
        self._header = values
        if self._template == 'column':
            field_names = values
            self.table.add_row(values)
        else:
            field_names = ['KEY', 'VALUE']
        self.table.field_names = field_names
        for i in range(len(field_names)):
            self.table.align[field_names[i]] = 'l'
            if self._column_widths:
                self.table.max_width[field_names[i]] = self._column_widths[i]

    def print(self):
        print(self.table)


@contextmanager
def profile_print(column_widths=None, field_format=None, template='column'):
    out_buffer = ProfilePrinter(column_widths, field_format, template)
    try:
        yield out_buffer
    finally:
        out_buffer.print()


def profile_timeit(fn_callable, repeat=1):
    ret = fn_callable()
    return ret, min(timeit.repeat(fn_callable, repeat=repeat, number=1))


def profile_cprofile(fn_callable, prof_file):
    prof = cProfile.Profile()
    ret = prof.runcall(fn_callable)
    prof.dump_stats(prof_file)
    prof_stats = StringIO()
    p = pstats.Stats(prof_file, stream=prof_stats)
    p.strip_dirs().sort_stats('cumulative').print_stats(0.5)
    return ret, prof_stats.getvalue()


class Profile(object):

    def __init__(self, tool, tool_cfg, fn_id):
        self.tool = tool
        self.tool_cfg = tool_cfg
        self.fn_id = fn_id

    def _set_decorator_params(self):
        if callable(self.tool):
            self.tool = self.tool()
        if callable(self.tool_cfg):
            self.tool_cfg = self.tool_cfg()

    def __call__(self, fn):

        def wrapped_fn(*args, **kwargs):
            self._set_decorator_params()
            fn_callable = functools.partial(fn, *args, **kwargs)
            if self.tool == 'timeit':
                return profile_timeit(fn_callable, **self.tool_cfg)
            elif self.tool == 'cprofile':
                prof_file = os.path.join(PROF_DIR, self.fn_id(*args, **kwargs))
                return profile_cprofile(fn_callable, prof_file=prof_file)
            else:
                raise ValueError('Invalid profiling tool specified: {}.'.format(self.tool))

        return wrapped_fn
