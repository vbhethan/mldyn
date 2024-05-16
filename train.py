"""
Training script for NRI model
"""

import time
import argparse
import pickle
import os
import datetime
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import wandb

from mldyn.layers.layers import MLPEncoder, MLPDecoder
from mldyn.utils import (
    encode_onehot,
    get_triu_indices,
    get_tril_indices,
    gumbel_softmax,
    my_softmax,
    nll_gaussian,
    kl_categorical_uniform,
)
from mldyn.data.dataloaders import load_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=100, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
parser.add_argument(
    "--learning_rate_decay",
    type=float,
    default=200,
    help="Learning rate decay, after how many epochs to decay by gamma.",
)
parser.add_argument(
    "--gamma", type=float, default=0.5, help="Learning rate decay factor."
)
parser.add_argument(
    "--encoder-hidden", type=int, default=256, help="Number of hidden units."
)
parser.add_argument(
    "--decoder-hidden", type=int, default=256, help="Number of hidden units."
)
parser.add_argument(
    "--temp", type=float, default=0.5, help="Temperature for Gumbel softmax."
)
parser.add_argument(
    "--num-atoms", type=int, default=5, help="Number of particles in simulation."
)
parser.add_argument(
    "--encoder-dropout",
    type=float,
    default=0.0,
    help="Dropout rate (1 - keep probability).",
)
parser.add_argument(
    "--decoder-dropout",
    type=float,
    default=0.0,
    help="Dropout rate (1 - keep probability).",
)
parser.add_argument(
    "--save-folder",
    type=str,
    default="logs",
    help="Where to save the trained model and other info.",
)
parser.add_argument(
    "--load-folder",
    type=str,
    default="",
    help="Where to load the trained model if finetunning. Leave empty to train from scratch",
)
parser.add_argument(
    "--edge-types", type=int, default=2, help="The number of edge types to infer."
)
parser.add_argument(
    "--dims",
    type=int,
    default=3,
    help="The number of input dimensions (e.g., position + velocity).",
)
parser.add_argument(
    "--timesteps", type=int, default=50, help="The number of time steps per sample."
)
parser.add_argument(
    "--prediction-steps",
    type=int,
    default=10,
    metavar="N",
    help="The number of prediction steps to include in loss",
)
parser.add_argument(
    "--skip-first",
    action="store_true",
    default=False,
    help="Skip first edge type in decoder",
)
parser.add_argument("--var", type=float, default=5e-5, help="Output variance.")
parser.add_argument(
    "--prior", action="store_true", default=False, help="Whether to use sparsity prior."
)
parser.add_argument(
    "--datafile-basename",
    type=str,
    default="data",
    help="The basename of the npy file containing the data.",
)
parser.add_argument(
    "--data-dir",
    type=str,
    default="./sim_data",
    help="The directory containing the data.",
)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")

if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = "{}/exp{}/".format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, "metadata.pkl")
    encoder_file = os.path.join(save_folder, "encoder.pt")
    decoder_file = os.path.join(save_folder, "decoder.pt")
    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, "w")

    pickle.dump({"args": args}, open(meta_file, "wb"))
else:
    print("NOTE: No save-folder provided! Will not save anything.")
    print("Testing will not work!")


train_loader, loc_max, loc_min = load_data(
    args.datafile_basename, batch_size=1, data_dir=args.data_dir
)

off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = torch.from_numpy(encode_onehot(np.where(off_diag)[0])).float()
rel_send = torch.from_numpy(encode_onehot(np.where(off_diag)[1])).float()

# instantiate the MLPEncoder
encoder = MLPEncoder(
    args.timesteps * args.dims,
    args.encoder_hidden,
    args.edge_types,
    args.encoder_dropout,
)


decoder = MLPDecoder(
    n_in_node=args.dims,
    edge_types=args.edge_types,
    msg_hid=args.decoder_hidden,
    msg_out=args.decoder_hidden,
    n_hid=args.decoder_hidden,
    do_prob=args.decoder_dropout,
    skip_first=args.skip_first,
)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, "encoder.pt")
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, "decoder.pt")
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr
)

scheduler = lr_scheduler.StepLR(
    optimizer, step_size=args.learning_rate_decay, gamma=args.gamma
)

triu_indices = get_triu_indices(args.num_atoms)
tril_indices = get_tril_indices(args.num_atoms)

if args.prior:
    raise NotImplementedError("Sparsity prior is not implemented yet.")

if args.cuda:
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    rel_rec = rel_rec.to(device)
    rel_send = rel_send.to(device)
    triu_indices = triu_indices.to(device)
    tril_indices = tril_indices.to(device)


def train(epoch):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()

    for batch_idx, example in enumerate(train_loader):
        data = torch.stack(list(example)).to(device)
        data = data.squeeze(0)

        optimizer.zero_grad()

        logits = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=True)
        prob = my_softmax(logits, -1)

        output = decoder(data, edges, rel_rec, rel_send, args.prediction_steps)

        target = data[:, :, 1:, :]
        # print("output shape", output.shape)
        # print("target shape", target.shape)
        loss_nll = nll_gaussian(output, target, args.var)

        if args.prior:
            raise NotImplementedError("Sparsity prior is not implemented yet.")
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        loss = loss_nll + loss_kl

        loss.backward()
        optimizer.step()
        scheduler.step()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

        print(
            f"Epoch: {epoch:04d}",
            f"nll_train: {np.mean(nll_train):.10f}",
            f"kl_train: {np.mean(kl_train):.10f}",
            f"mse_train: {np.mean(mse_train):.10f}",
            f"time: {time.time() - t:.4f}s",
        )

        if args.save_folder:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print(
                "Epoch: {:04d}".format(epoch),
                "nll_train: {:.10f}".format(np.mean(nll_train)),
                "kl_train: {:.10f}".format(np.mean(kl_train)),
                "mse_train: {:.10f}".format(np.mean(mse_train)),
                "time: {:.4f}s".format(time.time() - t),
                file=log,
            )
            log.flush()

    return np.mean(nll_train)


# Train model
t_total = time.time()
run = wandb.init(project="mldyn", entity="vbhethan")
for epoch in range(args.epochs):
    train_loss = train(epoch)
    print("Finished with epoch: ", epoch)
    run.log({"train/train_loss": train_loss, "epoch": epoch})

print("Optimization Finished!")

if args.save_folder:
    print("Finished with Model Training, saved to {}".format(save_folder))
    print("Epoch {}: Loss: {}".format(epoch, train_loss), file=log)
    log.flush()
