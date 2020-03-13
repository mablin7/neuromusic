#!/bin/sh
tensorboard --logdir=/storage/runs &
sshd -D
exec python neuromusic/main.py