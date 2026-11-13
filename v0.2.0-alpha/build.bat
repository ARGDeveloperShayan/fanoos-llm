@echo off
color 2
title training FanoosLLM v0.3b
python data/fanoosllm/prepare.py
python train.py config/fanoosllm.py
pause