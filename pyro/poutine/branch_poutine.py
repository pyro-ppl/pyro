import torch

from .poutine import Poutine


class BranchPoutine(Poutine):
    """
    Branch on discrete choices, expanding one batch dim per sample statement.

    This is essentially a vectorized version of poutine.iter_discrete_traces.
    Whereas iter_discrete_traces iterates over branches, this poutine adds a
    batch dimension to tensors sampled at each discrete sample statement,
    and scales each batch element of that tensor by local probability.
    """

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"]:
            return msg["ret"]

        # Sample as usual if fn is not enumerable (e.g. is continuous).
        fn = msg["fn"]
        if not getattr(fn, "enumerable", False):
            return super(BranchPoutine, self)._pyro_sample(self, msg)

        # Expand val by one batch dim and set a vectorized scale.
        args, kwargs = msg["args"], msg["kwargs"]
        val = torch.stack(fn.support(*args, **kwargs))
        log_pdf = fn.batch_log_pdf(val, *args, **kwargs)
        msg["scale"] = msg["scale"] * torch.exp(log_pdf)

        msg["done"] = True
        return val
