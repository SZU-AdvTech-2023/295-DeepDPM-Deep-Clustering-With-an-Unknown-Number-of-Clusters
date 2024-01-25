import argparse
import torch
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel
from src.datasets import CustomDataset

# LOAD MODEL FROM CHECKPOINT
cp_path = "./saved_models/USPS/default_exp/epoch=499-step=36499.ckpt" # E.g.: "./saved_models/USPS/default_exp/epoch=499-step=36499.ckpt"
cp_state = torch.load(cp_path)
data_dim =10 # E.g. for MNIST, it would be 10 if the network was trained on the embeedings supplied, or 28*28 otherwise.
K = cp_state['state_dict']['cluster_net.class_fc2.weight'].shape[0] 
hyper_param = cp_state['hyper_parameters']
args = argparse.Namespace()
for key, value in hyper_param.items():
    setattr(args, key, value)

model = ClusterNetModel.load_from_checkpoint(
    checkpoint_path=cp_path,
    input_dim=data_dim,
    init_k = K,
    hparams=args
    )

# Example for inference :
model.eval()
dataset_obj = CustomDataset(args)


train_loader, val_loader = dataset_obj.get_loaders()


# cluster_assignments = []
# for data, label in train_loader:
#     soft_assign = model(data)
#     hard_assign = soft_assign.argmax(-1)
#     cluster_assignments.append(hard_assign)
# print(cluster_assignments)


