#!/bin/sh
mkdir -p /storage/runs
tensorboard --logdir=/storage/runs --host 0.0.0.0 &
sshd -D
exec python neuromusic/main.py