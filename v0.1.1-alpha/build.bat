@echo off
color 2
title training TinyLLM v0.2a
python generate_data.py
python train.py
pause