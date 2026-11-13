@echo off
color 2
title training FanoosLLM v0.5a
python data/prepare.py
python train.py config/config.py
pause