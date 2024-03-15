import os

real_world_testpath = '/beegfs/project/bmbf-need/spectral-analysis/cnn-spectral-analysis/data/database_autoencoder/real_world_measured/'

# Loop through each subdirectory in the real_world_testpath
for subdir in os.listdir(real_world_testpath):
    print(subdir)