class WelfordOnlineStatistics(object):
    def __init__(self):
        self.sites = {}

    def update_statistics(self, z):
        for site_name, site_value in z.items():
            n_samples, mean, m2 = self.sites.get(site_name, (0, 0., 0.))
            n_samples += 1
            prev_diff = site_value - mean
            mean += prev_diff / n_samples
            cur_diff = site_value - mean
            m2 += prev_diff * cur_diff
            self.sites[site_name] = (n_samples, mean, m2)

    def get_variances(self):
        variances = {}
        for site_name, site_value in self.sites.items():
            n_samples, mean, m2 = self.sites[site_name]
            if n_samples < 2:
                raise RuntimeError('Insufficient samples to compute variance for site: "{}"'.format(site_name))
            variances[site_name] = m2 / (n_samples - 1)
        return variances
