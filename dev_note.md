# Pre-commit

- If you install QoA4ML from source, install the dev dependency group
  (defined under ``[dependency-groups]`` in ``pyproject.toml``) with uv:

  ```bash
  # uv-managed (preferred; matches CI and the committed uv.lock)
  uv sync --group dev

  # or, non-uv: install dev tools explicitly from pyproject.toml
  # (see the ``dev`` block under ``[dependency-groups]``)
  ```

  ``pip install ".[dev]"`` will **not** work because ``dev`` is not listed
  in ``[project.optional-dependencies]``.

- Installing the dev group pulls in pre-commit; activate the git hooks:

  ```bash
  pre-commit install
  ```

- Pre-commit cannot automatically re-stage auto-formatted files, so if a
  hook rewrites a file you must ``git add`` it manually and re-run.
- Run pre-commit across the whole repo, not just staged files:

  ```bash
  pre-commit run --all-files
  ```
