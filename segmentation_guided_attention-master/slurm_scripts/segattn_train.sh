#!/bin/bash
#SBATCH --job-name=segattn-rcnn       # Nombre del trabajo
#SBATCH --output=output/segattn_wealthy_%j.log         # Nombre del output (%j se reemplaza por el ID del trabajo
#SBATCH --error=output/err/segattn_wealthy_%j.err          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=4            # Numero de cores por tarea
#SBATCH --distribution=cyclic:cyclic # Distribuir las tareas de modo ciclico
#SBATCH --time=7-00:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --mem-per-cpu=10000mb         # Memoria por proceso
#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=afcadiz@uc.cl    # El mail del usuario
#SBATCH --partition=ialab-high        # Se tiene que elegir una partición de nodos con GPU
#SBATCH --gres=gpu:1080Ti:1       # Usar 2 GPUs (se pueden usar N GPUs de marca especifica de la manera --gres=gpu:marca:N)
#SBATCH --dependency=afterok:500

export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDADIR=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64

pyenv/bin/python3 train.py  --model segattn \
--max_epochs 30 \
--premodel resnet \
--attribute wealthy \
--wd 0 \
--lr 0.001  \
--batch_size 4 \
--dataset ../datasets/placepulse  \
--model_dir ../storage/models_seg  \
--tag large_images \
--csv votes/ \
--attention_normalize local \
--n_layers 1 --n_heads 1 --n_outputs 1 \
--eq --cuda \
--cm \
--softmax \
--pbar \
--ft 
