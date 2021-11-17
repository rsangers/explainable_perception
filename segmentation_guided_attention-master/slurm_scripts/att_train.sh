#!/bin/bash
#SBATCH --job-name=attention-rcnn       # Nombre del trabajo
#SBATCH --output=output/attention_beautiful_%j.log         # Nombre del output (%j se reemplaza por el ID del trabajo
#SBATCH --error=output/err/attention_beautiful_%j.err          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=4            # Numero de cores por tarea
#SBATCH --distribution=cyclic:cyclic # Distribuir las tareas de modo ciclico
#SBATCH --time=7-00:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --mem-per-cpu=8000mb         # Memoria por proceso
#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=afcadiz@uc.cl    # El mail del usuario
#SBATCH --partition=ialab-high        # Se tiene que elegir una partición de nodos con GPU
#SBATCH --gres=gpu:1080Ti:1       # Usar 2 GPUs (se pueden usar N GPUs de marca especifica de la manera --gres=gpu:marca:N)
#SBATCH --dependency=afterok:362


pyenv/bin/python3 train.py  --model attentionrcnn \
--max_epochs 10 \
--premodel resnet \
--attribute beautiful \
--wd 0.00001 \
--lr 0.001  \
--batch_size 32 \
--dataset ../datasets/placepulse  \
--model_dir ../storage/models_seg  \
--tag attn_resnet_drop \
--csv votes/ \
--attention_normalize local \
--n_layers 1 --n_heads 1 --n_outputs 1 \
--eq --cuda \
--cm \
--sgd \
--lr_decay \
--ft 
