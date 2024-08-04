<!--
Copyright Contributors to the Pyro project.

SPDX-License-Identifier: Apache-2.0
-->

# Documentation #
Pyro Documentation is supported by [Sphinx](http://www.sphinx-doc.org/en/stable/). 
To build the docs, run from the toplevel directory:
```
make docs
```

## Installation ##
```
pip install -r requirements.txt
```

Note that you will need to install [graphviz](https://www.graphviz.org/) separately.

## Workflow ##
To change the documentation, update the `*.rst` files in `source`.

To build the docstrings, `sphinx-apidoc [options] -o <output_path> <module_path> [exclude_pattern, ...]`

To build the html pages, `make html`
