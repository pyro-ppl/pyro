# Pyro release management

This describes the process by which versions of Pyro are officially released to the public.

## Versioning

Releases are versioned according to the `version_prefix` constant in [pyro/__init__.py](pyro/__init__.py).
Pyro releases follow semantic versioning with the following caveats:

- Behavior of documented APIs will remain stable across minor releases, except for bug fixes and features marked EXPERIMENTAL or DEPRECATED.
- Serialization formats will remain stable across patch releases, but may change across minor releases (e.g. if you save a model in 1.0.0, it will be safe to load it in 1.0.1, but not in 1.1.0).
- Undocumented APIs, features marked EXPERIMENTAL or DEPRECATED, and anything in `pyro.contrib` may change at any time (though we aim for stability).
- All deprecated features throw a `FutureWarning` and specify possible work-arounds. Features marked as deprecated will not be maintained, and are likely to be removed in a future release.
- If you want more stability for a particular feature, [contribute](https://github.com/pyro-ppl/pyro/blob/dev/CONTRIBUTING.md) a unit test.

## Release process

Pyro is released at irregular cadence, typically about 4 times per year.

Releases are managed by:
- [Neeraj Pradhan](https://github.com/neerajprad) npradhan@uber.com
- [Fritz Obermeyer](https://github.com/fritzo) fritzo@uber.com
- [JP Chen](https://github.com/jpchen)

Releases and release notes are published to [github](https://github.com/pyro-ppl/pyro/releases).
Documentation for is published to [readthedocs](https://docs.pyro.ai).
Release builds are published to [pypi](https://pypi.org/project/pyro-ppl/).
