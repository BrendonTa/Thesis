For usage  
git clone https://github.com/BrendonTa/Thesis.git

Required installs are:
Gymnasium
pip install Gymnasium
Mujoco
mkdir ~/.mujoco && mv /mnt/c/Users/<you>/Downloads/mujoco313-linux-x86_64.tar.gz ~/.mujoco && pushd ~/.mujoco && tar xf mujoco313-linux-x86_64.tar.gz && popd
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Path_to/.mujoco/mujoco313/bin
pip install Mujoco == 2.3.7
Stable-Baselines 3
pip install stable-baselines3[extra]
