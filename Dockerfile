FROM python:3.10-slim

# Install Slurm, Munge, and system tools
RUN apt-get update && apt-get install -y \
    slurm-wlm \
    munge \
    iputils-ping \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- FIX START ---
# Create Munge Key manually using dd
# Munge requires the key to be owned by 'munge' and have 400 permissions
RUN mkdir -p /etc/munge \
    && dd if=/dev/urandom of=/etc/munge/munge.key bs=1 count=1024 \
    && chmod 400 /etc/munge/munge.key \
    && chown munge:munge /etc/munge/munge.key
# --- FIX END ---

# Install PyTorch (CPU version for smaller image size)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Setup directories and permissions
RUN mkdir -p /var/log/slurm /var/run/slurm /data
RUN chown -R slurm:slurm /var/log/slurm /var/run/slurm

# Copy Entrypoint
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]