# # just for testing
# PYTHONPATH=$(pwd) python3 models/vilt_peft.py --sample --num_epochs=100

# pretrain
PYTHONPATH=$(pwd) python3 models/vilt_peft.py --num_epochs=100 --batch_size=64