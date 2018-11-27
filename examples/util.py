VERSION = '0.3'


def check_pyro_version():
    """
    Check Pyro dependency compatibility
    """
    import pyro
    version = pyro.__version__.split('.')
    ex_version = VERSION.split('.')
    major, minor = version[0], version[1]
    ex_major, ex_minor = ex_version[0], ex_version[1]
    if ex_major != major or ex_minor != minor:
        raise ImportError('Pyro examples version: {}.{} does not match '
                          'the Pyro version: {}.{}.  Try updating Pyro '
                          'to a matching version.'
                          .format(ex_major, ex_minor, major, minor))
