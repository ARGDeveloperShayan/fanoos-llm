@echo off
color 2
title training TinyLLM v0.3b
python data/tinyllm/prepare.py
python train.py config/tinyllm.py
pause