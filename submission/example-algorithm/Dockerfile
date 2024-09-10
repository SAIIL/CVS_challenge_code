FROM --platform=linux/amd64 pytorch/pytorch
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# You can add additional helper function scripts or directories but must add them in here and set permission

# Helper script example
#COPY --chown=user:user helpers.py /opt/app/

# Helper directory example
#COPY --chown=user:user util /opt/app/util

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

# Note that the container entrypoint is the inference.py script
ENTRYPOINT ["python", "inference.py"]
