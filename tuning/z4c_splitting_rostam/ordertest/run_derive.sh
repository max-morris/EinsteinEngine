#!/bin/bash
#SBATCH -p medusa
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH -J z4c-derive-order
cd /home/sbrandt/repos/EinsteinEngine/tuning/z4c_splitting_rostam/ordertest
export PYTHONPATH=/home/sbrandt/repos/EinsteinEngine
/home/sbrandt/repos/EinsteinEngine/venv/bin/python -u derive_order.py
