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

Citations for codes 
R.D Lazcano, K. Andreas, J.J Tai, S.R Lee, J Terry, “Gymnasium-Robotics” github.com. https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/README.md (Accessed: Oct. 20, 2023) 
G.Yang, “gym-fetch”, github.com. https://github.com/geyang/gymfetch.git (Accessed: Oct. 20, 2023)
wangcongrobot, “dual_ur5_husky_mujoco”, github.com. https://github.com/wangcongrobot/dual_ur5_husky_mujoco.git  (Accessed: Oct. 20, 2023) 
