

# https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
# unbounded memoize
# alternate in py3: https://docs.python.org/3/library/functools.html
# lru_cache
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]
