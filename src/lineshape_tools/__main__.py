import logging
from typing import Annotated

import cyclopts
from cyclopts import Group, Parameter

from lineshape_tools.cli import (
    analyze_dynmat,
    collect,
    compute_dynmat,
    compute_lineshape,
    convert_from_phonopy,
    gen_confs,
    gen_ft_config,
)

app = cyclopts.App(help_format="md")
app.meta.group_parameters = Group("Commands")

for func in (
    analyze_dynmat,
    collect,
    compute_dynmat,
    compute_lineshape,
    convert_from_phonopy,
    gen_confs,
    gen_ft_config,
):
    app.command()(func)


@app.meta.default
def _app_launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[bool, Parameter(negative="--quiet")] = False,
) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s" if verbose else "[*] %(message)s"
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(level=log_level, format=fmt, encoding="utf-8")

    # remove logging from imported libraries, wow this is a monster of a line
    logging.getLogger().handlers[0].addFilter(logging.Filter(name=__name__.split(".")[0]))

    logging.getLogger(__name__).debug("starting application")
    app(tokens)


if __name__ == "__main__":
    app.meta()
