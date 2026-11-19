"""Architecture tests: layering rules as build-time facts.

Tonight's rule, from Jerry's own argument about AI-generated code:
review opinions should become checks, not comments repeated forever.
If generated (or hand-written) code breaks the layering, this fails
instead of silently accreting a new dependency direction.

Rules, matching the v1 layout in harness/README.md:
- core/      — pure entities, stdlib only. No harness package imports.
- usecases/ — runner, ports, gates, replay. May import core and
               usecases only — never adapters, never app.
- adapters/ — concrete implementations. May import core and usecases
               only — never app (the composition root must not be
               pulled in from below).
- app/      — the composition root (Harness.define). It alone may wire
               adapters + core + usecases together.

This scans source with ast, so it catches an import even if the
module is never executed in the other tests. Tests and examples are
not scanned: tests legitimately import everything, and examples are
consumers, not layers.

Run:  python3 harness/tests/test_architecture.py
"""

import ast
import os
import sys

HARNESS_DIR = os.path.join(os.path.dirname(__file__), "..")
HARNESS_PACKAGES = {"core", "usecases", "adapters", "app"}
STDLIB = set(getattr(sys, "stdlib_module_names", set())) | {"__future__"}

# layer -> harness packages it may import (stdlib always allowed)
ALLOWED = {
    "core": set(),
    "usecases": {"core", "usecases"},
    "adapters": {"core", "usecases"},
}


def top_level(module):
    return (module or "").split(".")[0]


def imports_of(path):
    tree = ast.parse(open(path).read())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level:
            # Relative imports inside harness are intra-package; the
            # containing package is the layer, handled by the caller.
            pass
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def check_layer(layer):
    layer_dir = os.path.join(HARNESS_DIR, layer)
    violations = []
    checked = 0
    for root, _dirs, files in os.walk(layer_dir):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            checked += 1
            for module in imports_of(path):
                top = top_level(module)
                if top in HARNESS_PACKAGES and top not in ALLOWED[layer]:
                    violations.append(
                        "%s imports %s (layer %s may not import %s)"
                        % (os.path.relpath(path, HARNESS_DIR), module, layer, top)
                    )
                elif top not in HARNESS_PACKAGES and top not in STDLIB:
                    # Third-party import in a layer that should be stdlib
                    # + harness only (e.g. openai belongs in adapters'
                    # model adapter, which is exempted below).
                    if not (layer == "adapters" and top == "openai"):
                        violations.append(
                            "%s imports non-stdlib %s" % (
                                os.path.relpath(path, HARNESS_DIR), module)
                        )
    return checked, violations


def main():
    total = 0
    failures = []
    for layer in ("core", "usecases", "adapters"):
        checked, violations = check_layer(layer)
        total += checked
        failures.extend(violations)
        print("%s: %d files, %d violations" % (layer, checked, len(violations)))
    assert not failures, "\n".join(failures)
    print("architecture: %d files checked, layering holds" % total)


if __name__ == "__main__":
    main()
