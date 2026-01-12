#!/bin/bash
set -e

# --- CRITICAL FIX FOR DOCKER ---
# 1. Mask DBUS to prevent Slurm from trying to use systemd
export DBUS_SYSTEM_BUS_ADDRESS=unix:path=/dev/null
export DBUS_SESSION_BUS_ADDRESS=unix:path=/dev/null

# 2. Remove Systemd/DBus directories if they exist
if [ -d /run/systemd ]; then
    rm -rf /run/systemd
fi
if [ -d /run/dbus ]; then
    rm -rf /run/dbus
fi
# -------------------------------

# Start Munge
service munge start

# Fix Shared Volume Permissions
chown -R root:root /data

if [ "$HOSTNAME" == "node-1" ]; then
    echo "--- Starting Node-1 (Controller + Worker) ---"
    echo "Starting slurmctld..."
    /usr/sbin/slurmctld
    
    echo "Starting slurmd..."
    /usr/sbin/slurmd
    
    # Keep container alive
    tail -f /var/log/slurm/slurmctld.log
else
    echo "--- Starting Node-2 (Worker Only) ---"
    sleep 2
    /usr/sbin/slurmd -D
fi