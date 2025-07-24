import sys

import torch
from absl import flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 1024, help="Batch size to compute FID")

FLAGS(sys.argv)


# Define the model
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device = torch.device("cpu")

new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)


# Load the model
PATH = f"{FLAGS.input_dir}/{FLAGS.model}/{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"]
# state_dict = checkpoint["net_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()


# Define the integration method if euler is used
if FLAGS.integration_method == "euler":
    node = NeuralODE(new_net, solver=FLAGS.integration_method)


def straightness(batch_size, num_gen, integration_method, integration_steps, tol, device):
    norms = []
    n_batches = (num_gen + batch_size - 1) // batch_size
    for _ in range(n_batches):
        with torch.no_grad():
            x = torch.randn(batch_size, 3, 32, 32, device=device)
            if integration_method == "euler":
                print("Use method: ", integration_method)
                t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
                traj = node.trajectory(x, t_span=t_span)
            else:
                print("Use method: ", integration_method)
                t_span = torch.linspace(0, 1, 2, device=device)
                traj = odeint(
                    new_net, x, t_span, rtol=tol, atol=tol, method=integration_method
                )
        traj = torch.stack((x, traj), dim=0) # (integration_steps + 1, batch_size, 3, 32, 32)
        displacements = traj[1:] - traj[:-1] # (integration_steps, batch_size, 3, 32, 32)
        delta_t = t_span[1:] - t_span[:-1]  # (integration_steps)
        velocities = displacements / delta_t.view(-1, 1, 1, 1, 1)  # (integration_steps, batch_size, 3, 32, 32)
        x1_minus_x0 = traj[-1] - traj[0] # (batch_size, 3, 32, 32)
        sq_norm = torch.linalg.norm(
            x1_minus_x0.view(1, batch_size, -1) - velocities.view(integration_steps, batch_size, -1), 
            dim=-1
        ) ** 2
        norms.append(sq_norm.cpu())
    norms = torch.cat(norms, dim=1)[:, :num_gen]  # (integration_steps, num_gen)
    return torch.mean(norms)

def add_nfe_counter(model):
    model.nfe = 0
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        model.nfe += 1
        return original_forward(*args, **kwargs)

    model.forward = wrapped_forward

add_nfe_counter(new_net)

print("Start computing straightness")
score = straightness(
    batch_size=FLAGS.batch_size_fid,
    num_gen=FLAGS.num_gen,
    integration_method=FLAGS.integration_method,
    integration_steps=FLAGS.integration_steps,
    tol=FLAGS.tol,
    device=device
)
print()
print("Total NFE: ", new_net.nfe)
print("Straightness: ", score)
