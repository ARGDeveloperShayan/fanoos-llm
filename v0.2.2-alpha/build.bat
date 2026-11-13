@echo off
color 2
title training FanoosLLM v0.4b
python data/prepare.py
python train.py config/config.py
pause