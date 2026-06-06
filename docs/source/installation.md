# Installation

## Linux/MacOS

### Prerequisites

- git
- Python 3.11 or higher
- [acados dependencies](https://docs.acados.org/installation/index.html)

Clone the repository and recursively update submodules:

```bash
git clone https://github.com/leap-c/leap-c.git
cd leap-c
git submodule update --init --recursive
```

### Python

Create a virtual environment and activate it. Assuming a required Python version of 3.11 or newer, you can use the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The following steps assume that the virtual environment is activated.

#### acados

Then change into the acados directory

```bash
cd external/acados
```

and build it as described in the [acados documentation](https://docs.acados.org/installation/index.html).
When running the `cmake` command, include the options `-DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=1`.
Afterwards, install the [python interface](https://docs.acados.org/python_interface/index.html) of acados.

#### PyTorch

Install PyTorch as described on the [PyTorch website](https://pytorch.org/get-started/locally/).

To install CPU-only PyTorch you can use:

``` bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### Install leap-c

To install the package containing minimum dependencies in the root directory of the repository, run:

```bash
pip install -e .
```

For also enabling rendering in some of our examples use:

```bash
pip install -e ".[rendering]"
```

For development, you might want to install all additional dependencies:

```bash
pip install -e ".[dev]"
```

See the [pyproject.toml](https://github.com/leap-c/leap-c/blob/main/pyproject.toml) for more information on package configurations.

### Using uv (alternative)

If you use [uv](https://docs.astral.sh/uv/), a single `uv sync` from the repository
root sets up the environment, including the `i4b` example (installed editable from the
`external/i4b` submodule together with its extra runtime dependencies) and the acados
Python interface (`acados_template`, installed editable from the `external/acados`
submodule):

```bash
git submodule update --init --recursive   # uv cannot initialize submodules itself
uv sync
```

> **Note:** `external/i4b` and `external/acados` must be checked out before running
> `uv sync`, otherwise the sync fails because the `i4b` workspace member and the
> `acados_template` path source cannot be built. To skip either, run
> `uv sync --no-group i4b` and/or `uv sync --no-group acados`.

> **Note:** `uv sync` only installs the *Python* package `acados_template`, which is
> enough for `import acados_template`. Running a solver additionally requires building the
> acados **C library** as described in the [acados](#acados) section above.

### Troubleshooting

In the [troubleshooting tab](https://leap-c.github.io/leap-c/troubleshooting.html),
we highlight how to fix common problems arising while using leap-c with VS Code.

## Windows

We recommend to use [WSL (Windows Subsystem for Linux)](https://ubuntu.com/desktop/wsl) and then following the guide above.
You can then conveniently program on your WSL, e.g.,
by using VS Code on Windows together with the "Remote Development" extension pack.

Note the [installation instructions for acados regarding WSL](https://docs.acados.org/installation/index.html#windows-10-wsl).
Also note the troubleshooting section for `plt.show()` in WSL below.

## Testing

To run the tests, use:

```bash
pytest tests -vv -s
```

## Linting and Formatting

Only relevant if you want to contribute to the repository.
For keeping our code style and our diffs consistent we use the [Ruff](https://docs.astral.sh/ruff/) linter and formatter.

To make this as easy as possible we also provide a [pre-commit](https://pre-commit.com/) config for running the linter and formatter automatically at every commit. For enabling pre-commit follow these steps:

1. Install pre-commit using pip (already done if you installed the additional "[dev]" dependencies of leap-c).

```bash
pip install pre-commit
```

2. In the leap-c root directory run

```bash
pre-commit install
```

Done! Now every commit will automatically be linted and formatted.