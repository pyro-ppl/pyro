# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get new version.
with open(os.path.join(root, "pyproject.toml")) as f:
    match = re.search(r'^version = "([^"]+)"', f.read(), re.MULTILINE)
if not match:
    raise RuntimeError("Unable to find project.version in pyproject.toml")
new_version = '"{}"'.format(match.group(1))

# Collect potential files.
filenames = []
for path in ["examples", "tutorial/source"]:
    for ext in ["*.py", "*.ipynb"]:
        filenames.extend(glob.glob(os.path.join(root, path, "**", ext), recursive=True))
filenames.sort()

# Update version string.
pattern1 = re.compile('assert pyro.__version__.startswith\\("[^"]*"\\)')
pattern2 = re.compile("assert pyro.__version__.startswith\\('[^']*'\\)")
text1 = f"assert pyro.__version__.startswith({new_version})"
text2 = text1.replace('"', "'")
for filename in filenames:
    with open(filename) as f:
        old_text = f.read()
    new_text = pattern1.sub(text1, old_text)
    new_text = pattern2.sub(text2, new_text)
    if new_text != old_text:
        print("updating {}".format(filename))
    with open(filename, "w") as f:
        f.write(new_text)
