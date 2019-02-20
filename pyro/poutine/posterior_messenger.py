from .replay_messenger import ReplayMessenger


class SamplePosteriorMessenger(ReplayMessenger):
    # This acts like ReplayMessenger but additionally replays cond_indep_stack.

    def _pyro_sample(self, msg):
        if msg["infer"].get("enumerate") == "parallel":
            super(SamplePosteriorMessenger, self)._pyro_sample(msg)
        if msg["name"] in self.trace:
            msg["cond_indep_stack"] = self.trace.nodes[msg["name"]]["cond_indep_stack"]
        super(SamplePosteriorMessenger, self)._pyro_sample(msg)
