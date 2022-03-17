module load python/3.7
module load cuda/11.0
module load cudnn/8.0.3

. configure.sh
virtualenv --no-download $MNET_VENV
source $MNET_VENV/bin/activate
pip install --no-index --upgrade pip
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index opencv_python_headless
pip install -U 'https://files.pythonhosted.org/packages/6b/68/2bacb80e13c4084dfc37fec8f17706a1de4c248157561ff33e463399c4f5/fvcore-0.1.3.post20210317.tar.gz'
pip install --no-index detectron2
pip install h5py