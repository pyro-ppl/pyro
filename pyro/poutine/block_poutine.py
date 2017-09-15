from .poutine import Poutine


class BlockPoutine(Poutine):
    """
    Blocks some things
    """

    def __init__(self, fn,
                 hide_all=True, expose_all=False,
                 hide=None, expose=None,
                 hide_types=None, expose_types=None):
        """
        Constructor for blocking poutine
        Default behavior: block everything
        """
        super(BlockPoutine, self).__init__(fn)
        # first, some sanity checks:
        # hide_all and expose_all intersect?
        assert (hide_all is False and expose_all is False) or \
               (hide_all != expose_all), "cannot hide and expose a site"

        # hide and expose intersect?
        if hide is None:
            hide = []
        else:
            hide_all = False

        if expose is None:
            expose = []
        assert set(hide).isdisjoint(set(expose)), \
            "cannot hide and expose a site"

        # hide_types and expose_types intersect?
        if hide_types is None:
            hide_types = []
        if expose_types is None:
            expose_types = []
        assert set(hide_types).isdisjoint(set(expose_types)), \
            "cannot hide and expose a site type"

        # now set stuff
        self.hide_all = hide_all
        self.expose_all = expose_all
        self.hide = hide
        self.expose = expose
        self.hide_types = hide_types
        self.expose_types = expose_types

    def _block_up(self, msg):
        """
        A stack-blocking operation
        """
        # hiding
        if (msg["name"] in self.hide) or \
           (msg["type"] in self.hide_types) or \
           ((msg["name"] not in self.expose) and (msg["type"] not in self.expose_types) and self.hide_all):
            return True
        # otherwise expose
        else:
            return False
