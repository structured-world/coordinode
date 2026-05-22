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
    pip install --no-cache-dir numpy && \
    rm -rf /tmp/wheels

# Smoke-import so a broken build fails at docker build time, not at first run.
RUN python -c "from coordinode_embedded import Hnsw; print('coordinode_embedded.Hnsw OK')"
