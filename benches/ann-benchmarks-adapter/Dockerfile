# ann-benchmarks adapter for CoordiNode.
#
# Two install modes, controlled at build time:
#
#   1. Pre-release wheel (the bench-CI default).  The orchestrator script
#      runs `maturin build` against the coordinode SHA being benched and
#      drops the resulting wheel beside this Dockerfile before
#      `docker build`.  `COPY *.whl` then picks it up.  This is how every
#      push to coordinode/main gets its own bench number.
#
#   2. PyPI release.  When `--build-arg PYPI_VERSION=<ver>` is passed, we
#      `pip install coordinode-embedded==<ver>` instead of the local wheel.
#      This is what the eventual upstream PR to erikbern/ann-benchmarks
#      will use (no submodule magic, fully reproducible from public
#      PyPI artifacts).
#
# Both modes pull `numpy` because ann-benchmarks's BaseANN harness passes
# arrays without going through a higher-level dataset-loading layer.

FROM ann-benchmarks

ARG PYPI_VERSION=

# Upstream ann-benchmarks base is Ubuntu 22.04 / Python 3.10.
# coordinode-embedded is pyo3 abi3-py311 (requires ≥3.11), so we pull
# 3.11 from deadsnakes, switch the `python` / `python3` alias to it,
# and re-install the ann-benchmarks runtime requirements for 3.11 so
# `python -u run_algorithm.py` (the image's ENTRYPOINT) keeps working.
# Only the coordinode adapter pays this cost; other algos stay on the
# untouched base.
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 python3.11-distutils python3.11-venv curl ca-certificates && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    apt-get clean

# Re-install ann-benchmarks runtime deps for Python 3.11 (the base
# image installed them for 3.10 only). Match the upstream
# requirements.txt that was COPY'd onto the base image.
RUN pip install --no-cache-dir -r /home/app/requirements.txt

COPY *.whl /tmp/wheels/

RUN if [ -n "$PYPI_VERSION" ]; then \
      echo "Installing coordinode-embedded==${PYPI_VERSION} from PyPI"; \
      pip install --no-cache-dir "coordinode-embedded==${PYPI_VERSION}"; \
    else \
      WHEEL=$(ls /tmp/wheels/coordinode_embedded-*.whl 2>/dev/null | head -1); \
      if [ -z "$WHEEL" ]; then \
        echo "ERROR: neither PYPI_VERSION arg nor a local wheel under /tmp/wheels/ provided"; \
        exit 1; \
      fi; \
      echo "Installing local pre-release wheel $WHEEL"; \
      pip install --no-cache-dir "$WHEEL"; \
    fi && \
    rm -rf /tmp/wheels

# Smoke-import so a broken build fails at docker build time, not at first run.
RUN python -c "from coordinode_embedded import Hnsw; print('coordinode_embedded.Hnsw OK')"
