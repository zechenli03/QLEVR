#!/bin/bash
python gen_2d.py --num_images 50 && blender --background --python gen_3d.py
