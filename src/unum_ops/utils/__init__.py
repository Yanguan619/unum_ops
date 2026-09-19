import os


def find_ops_lib(env_var: str, so_name: str, so_rel: str, caller_file: str) -> str:
    env = os.environ.get(env_var)
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"{env_var} does not exist: {env}")
        return env
    here = os.path.dirname(os.path.abspath(caller_file))
    pkg_lib = os.path.join(here, "_libs", so_name)
    if os.path.exists(pkg_lib):
        return pkg_lib
    for _ in range(6):
        cand = os.path.join(here, so_rel)
        if os.path.exists(cand):
            return cand
        here = os.path.dirname(here)
    raise FileNotFoundError(
        f"{so_name} not found. Build it via the operator's CMakeLists or set "
        f"{env_var}=<path>"
    )