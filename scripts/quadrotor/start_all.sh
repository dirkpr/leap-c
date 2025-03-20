#!/bin/bash
tmux new-session -d -s training_session
tmux send-keys "conda activate leapc" C-m
tmux send-keys "python quadrotor_weights_sac.py" C-m
tmux split-window -v
tmux send-keys "conda activate leapc" C-m
tmux send-keys "python quadrotor_weights_sac_fop.py" C-m
tmux split-window -v
tmux send-keys "conda activate leapc" C-m
tmux send-keys "python quadrotor_weights_sac_zop.py" C-m
tmux attach -t training_session