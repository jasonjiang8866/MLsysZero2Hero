# The build script resolves the base to an immutable registry digest first.
ARG PYTHON_BASE
FROM ${PYTHON_BASE} AS build
ARG SOURCE_SHA
WORKDIR /opt/mlsys
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --no-cache-dir uv==0.13.5
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python \
 && printf '%s\n' "${SOURCE_SHA}" > SOURCE_SHA

FROM ${PYTHON_BASE}
ARG SOURCE_SHA
LABEL org.opencontainers.image.source="https://github.com/jasonjiang8866/MLsysZero2Hero"
LABEL org.opencontainers.image.revision="${SOURCE_SHA}"
COPY --from=build /opt/mlsys/.venv /opt/mlsys/.venv
COPY --from=build /opt/mlsys/SOURCE_SHA /opt/mlsys/SOURCE_SHA
ENV PATH="/opt/mlsys/.venv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["python", "-m", "mlsys_lab.worker"]
