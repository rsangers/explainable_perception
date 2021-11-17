#!/bin/bash
#SBATCH --job-name=avg_attn      # Nombre del trabajo
#SBATCH --output=output/avg_attn_%j.log         # Nombre del output (%j se reemplaza por el ID del trabajo
#SBATCH --error=output/err/avg_attn_%j.err          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=4            # Numero de cores por tarea
#SBATCH --distribution=cyclic:cyclic # Distribuir las tareas de modo ciclico
#SBATCH --time=7-00:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --mem-per-cpu=6000mb         # Memoria por proceso
#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=afcadiz@uc.cl    # El mail del usuario
#SBATCH --partition=ialab-high        # Se tiene que elegir una partición de nodos con GPU
#SBATCH --gres=gpu:1080Ti:1       # Usar 2 GPUs (se pueden usar N GPUs de marca especifica de la manera --gres=gpu:marca:N


pyenv/bin/python3 average_attentions.py

