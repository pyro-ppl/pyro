VERSION = '0.3'


def check_compatible_version():
    """
    Check Pyro dependency compatibility
    """
    import pyro
    version = pyro.__version__.split('.')
    ex_version = VERSION.split('.')
    major, minor = version[0], version[1]
    ex_major, ex_minor = ex_version[0], ex_version[1]
    assert ex_major == major
    assert ex_minor == minor
