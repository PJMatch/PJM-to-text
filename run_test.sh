#!/bin/bash
# Please use python 3.10/11/9 doesn't work with 3.14
[ -d "venv" ] || python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mim install mmpose
python mmpose_test.py
