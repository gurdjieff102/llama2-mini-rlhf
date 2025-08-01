#!/bin/bash
python scripts/train_sft.py
python scripts/train_reward.py
python scripts/train_ppo.py
