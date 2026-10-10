import os
import re
import sys

# Strip /oh/glibc236 and PyInstaller's temporary bundle directory (_MEIPASS)
# from LD_LIBRARY_PATH so child processes use the task container's native libs:
_orig_ld = os.environ.get(
    "LD_LIBRARY_PATH_ORIG", os.environ.get("LD_LIBRARY_PATH", "")
)
_meipass = getattr(sys, "_MEIPASS", "")
_clean_ld_paths = [
    p
    for p in _orig_ld.split(":")
    if p
    and p != "/oh/glibc236"
    and (not _meipass or p != _meipass)
    and not p.startswith("/tmp/_MEI")
]
if _clean_ld_paths:
    os.environ["LD_LIBRARY_PATH"] = ":".join(_clean_ld_paths)
else:
    os.environ.pop("LD_LIBRARY_PATH", None)
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
