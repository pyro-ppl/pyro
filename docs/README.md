# Docuemtnation #
Pyro Documentation is supported by [Sphinx](http://www.sphinx-doc.org/en/stable/). 

## Installation ##
```
pip install -r requirements.txt
```

## Workflow ##
To change the documentation, update the `*.rst` files in `source`.

To build the docstrings, `sphinx-apidoc [options] -o <output_path> <module_path> [exclude_pattern, ...]`

To build the html pages, `make html`
