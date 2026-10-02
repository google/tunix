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

from openhands.agent_server.__main__ import main

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
