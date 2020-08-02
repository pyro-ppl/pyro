# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
blacklist = ["/build/", "/dist/", "/pyro/_version.py"]
file_types = [
    ("*.py", "# {}"),
    ("*.cpp", "// {}"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
dirty = []

for basename, comment in file_types:
    copyright_line = comment.format("Copyright Contributors to the Pyro project.\n")
    # See https://spdx.org/ids-how
    spdx_line = comment.format("SPDX-License-Identifier: Apache-2.0\n")

    filenames = glob.glob(os.path.join(root, "**", basename), recursive=True)
    filenames.sort()
    filenames = [
        filename
        for filename in filenames
        if not any(word in filename for word in blacklist)
    ]
    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()

        # Ignore empty files like __init__.py
        if all(line.isspace() for line in lines):
            continue

        # Ensure first few line are copyright notices.
        changed = False
        lineno = 0
        if not lines[lineno].startswith(comment.format("Copyright")):
            lines.insert(lineno, copyright_line)
            changed = True
        lineno += 1
        while lines[lineno].startswith(comment.format("Copyright")):
            lineno += 1

        # Ensure next line is an SPDX short identifier.
        if not lines[lineno].startswith(comment.format("SPDX-License-Identifier")):
            lines.insert(lineno, spdx_line)
            changed = True
        lineno += 1

        # Ensure next line is blank.
        if not lines[lineno].isspace():
            lines.insert(lineno, "\n")
            changed = True

        if not changed:
            continue

        if args.check:
            dirty.append(filename)
            continue

        with open(filename, "w") as f:
            f.write("".join(lines))

        print("updated {}".format(filename[len(root) + 1:]))

if dirty:
    print("The following files need license headers:\n{}"
          .format("\n".join(dirty)))
    print("Please run 'make license'")
    sys.exit(1)
