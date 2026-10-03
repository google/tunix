import os
import re
import sys

# Strip /oh/glibc236 from LD_LIBRARY_PATH so child processes use task container's glibc:
if "LD_LIBRARY_PATH" in os.environ:
    paths = [p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p != "/oh/glibc236"]
    if paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths)
    else:
        del os.environ["LD_LIBRARY_PATH"]
# Ensure the agent server only imports from its own packaged bundle/runtime,
# never from the user repository or current working directory in the sandbox:
cwd = os.path.abspath(os.getcwd())
sys.path = [
    p for p in sys.path
    if p not in ("", ".")
    and os.path.abspath(p) not in (cwd, "/testbed", "/workspace")
    and not os.path.abspath(p).startswith(("/testbed/", "/workspace/"))
]

from openhands.agent_server.__main__ import main

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
