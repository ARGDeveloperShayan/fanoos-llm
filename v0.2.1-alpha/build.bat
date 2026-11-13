@echo off
color 2
title training FanoosLLM v0.4a
python data/fanoosllm/prepare.py
python train.py config/fanoosllm.py
pause