# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import os.path

import numpy as np
from torch_ema import ExponentialMovingAverage
from residual_controller.bridge.networks.conditional_unet_1D_si import *

def indicator_function(condition):
    # Create a tensor of zeros with the same shape as the condition
    result = torch.zeros_like(condition, dtype=torch.float32)

    # Set the elements to 1 where the condition is satisfied
    result[condition] = 1.0

    return result


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


class StochasticInterpolants:
    def __init__(self, model_args=None):

        if model_args:
            self.load_model_args(model_args)
        else:
            print("init SI without model args")

    def load_model_args(self,model_args):

        self.interpolant_type = model_args['interpolant_type']
        self.gamma_type = model_args['gamma_type']
        self.epsilon_type = model_args['epsilon_type']
        self.prior_policy = model_args['prior_policy']

        self.d = model_args['beta_max']

        self.t_min = 0.001
        self.gamma_inv_max = 200.0

        self.net = None
        self.ema = None
        self.prior_model = None

        if 'sde_type' in model_args:
            self.sde_type = model_args['sde_type']
        else:
            self.sde_type = 'vs'

        print(f"INIT SI with model args:{model_args} ")

    def epsilon(self, t):
        if self.epsilon_type == 't(t-1)':
            return t * (1 - t)
        elif self.epsilon_type == '1-t':
            return (1 - t) * 1.0
        elif self.epsilon_type == '1-sqrt(t)':
            return 1 - torch.sqrt(t)
        elif self.epsilon_type == '1-t^2':
            return 1 - torch.pow(t, 2)
        elif self.epsilon_type == '0':
            return t * 0.0
        else:
            raise NotImplementedError

    def gamma(self, t):
        if self.gamma_type == '(2t(t-1))^0.5':
            return 1.4142 * torch.sqrt(t * (1 - t))
        elif self.gamma_type == '2^0.5*t(t-1)':
            return 1.4142 * t * (1 - t)
        elif self.gamma_type == '(1-t)^2(2t)^0.5':
            return 1.4142 * torch.pow((1 - t), 2.0) * torch.sqrt(t)
        else:
            raise NotImplementedError

    def gamma_der(self, t):
        if self.gamma_type == '(2t(t-1))^0.5':
            return (1 - 2 * t) / torch.sqrt(2 * (t - torch.pow(t, 2)) + 1e-4)
        if self.gamma_type == '2^0.5*t(t-1)':
            return 1.4142 * (1 - 2 * t)
        elif self.gamma_type == '(1-t)^2(2t)^0.5':
            return 1.4142 * (2 * (t - 1) * torch.sqrt(t) + torch.pow((1 - t), 2.0) / (2.0 * torch.sqrt(t + 1e-4)))
        else:
            raise NotImplementedError

    def gamma_inv(self, t):
        if self.gamma_type == '(2t(t-1))^0.5':
            return torch.clamp(1 / (1.4142 * torch.sqrt(t * (1 - t) + 1e-4)), 0.0, self.gamma_inv_max)
        elif self.gamma_type == '2^0.5*t(t-1)':
            return torch.clamp(1 / (1.4142 * t * (1 - t) + 1e-4), 0.0, self.gamma_inv_max)
        elif self.gamma_type == '(1-t)^2(2t)^0.5':
            return torch.clamp(1 / (1.4142 * torch.pow((1 - t), 2.0) * torch.sqrt(t) + 1e-4), 0.0, self.gamma_inv_max)
        else:
            raise NotImplementedError

    def interpolant(self, x0, x1, gamma, t):
        if self.interpolant_type == 'linear':
            z = self.d * torch.randn_like(x0).float().to(x0.device)

            xt = (1 - t) * x0 + t * x1 + gamma * z

        elif self.interpolant_type == 'reverse_power3':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = 1 - torch.pow(t, 3)
            w_x1 = torch.pow(t, 3)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'reverse_power4':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = 1 - torch.pow(t, 4)
            w_x1 = torch.pow(t, 4)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'power3':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = torch.pow((1 - t), 3)
            w_x1 = 1 - torch.pow((1 - t), 3)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'power4':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = torch.pow((1 - t), 4)
            w_x1 = 1 - torch.pow((1 - t), 4)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'gaussian_encode_decode':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = torch.pow(torch.cos(t * np.pi), 2) * indicator_function(t <= 0.5)
            w_x1 = torch.pow(torch.cos(t * np.pi), 2) * indicator_function(t > 0.5)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'reverse_linear':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = (1 - 2 * t) * indicator_function(t <= 0.5)
            w_x1 = 1 - (1 - 2 * t) * indicator_function(t <= 0.5)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        else:
            raise NotImplementedError
        return xt, z

    def interpolant_dev(self, x1, x0, t):
        if self.interpolant_type == 'linear':
            partial_t = (x1 - x0)
        elif self.interpolant_type == 'power3':
            batch, *xdim = x1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 3 * torch.pow(1 - t_reshape, 2) * (x1 - x0)
        elif self.interpolant_type == 'power4':
            batch, *xdim = x1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 4 * torch.pow(1 - t_reshape, 3) * (x1 - x0)
        elif self.interpolant_type == 'reverse_power3':
            batch, *xdim = x1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 3 * torch.pow(t_reshape, 2) * (x1 - x0)
        elif self.interpolant_type == 'reverse_power4':
            batch, *xdim = x1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 4 * torch.pow(t_reshape, 3) * (x1 - x0)
        elif self.interpolant_type == 'gaussian_encode_decode':
            batch, *xdim = x1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = -2 * torch.pi * torch.cos(torch.pi * t_reshape) * torch.sin(torch.pi * t_reshape) * indicator_function(t_reshape <= 0.5) * x0
            partial_t += -2 * torch.pi * torch.cos(torch.pi * t_reshape) * torch.sin(torch.pi * t_reshape) * indicator_function(t_reshape > 0.5) * x1
        elif self.interpolant_type == 'reverse_linear':
            batch, *xdim = x1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = -2 * indicator_function(t_reshape <= 0.5) * x0
            partial_t += 2 * indicator_function(t_reshape <= 0.5) * x1
        else:
            raise NotImplementedError

        return partial_t

    def velocity_loss(self, v_net, t, xt, x0, x1, cond):
        t = torch.clip(t, self.t_min, 1.0 - self.t_min)
        partial_t = self.interpolant_dev(x1=x1, x0=x0, t=t)
        v = v_net(xt, t, global_cond=cond)
        v_reshape = v.flatten(-2)
        partial_t_reshape = partial_t.flatten(-2)

        loss = 0.5 * torch.norm(v_reshape, dim=-1) ** 2 - torch.sum(partial_t_reshape * v_reshape, dim=-1)
        return torch.mean(loss)

    # Do the same for other loss functions
    def score_loss(self, s_net, t, x_t, z, cond):
        t = torch.clip(t, self.t_min, 1.0 - self.t_min)
        s = s_net(x_t, t, global_cond=cond)

        s_reshape = s.flatten(-2)
        z_reshape = z.flatten(-2)
        loss = (0.5 * torch.norm(s_reshape, dim=-1) ** 2 + torch.sum(z_reshape * s_reshape, dim=-1))
        return torch.mean(loss)

    def b_loss(self, b_net, t, xt, x0, x1, z, cond):
        t = torch.clip(t, self.t_min, 1.0 - self.t_min)
        partial_t = self.interpolant_dev(x1=x1, x0=x0, t=t)

        gamma_der = self.gamma_der(t)
        b = b_net(xt, t, global_cond=cond)
        b_reshape = b.flatten(-2)
        partial_t_reshape = partial_t.flatten(-2)

        batch, *xdim = b_reshape.shape
        gamma_der_reshape = unsqueeze_xdim(gamma_der, xdim)

        z_reshape = z.flatten(-2)

        loss = 0.5 * torch.norm(b_reshape, dim=-1) ** 2 - torch.sum((partial_t_reshape + gamma_der_reshape * z_reshape) * b_reshape, dim=-1)
        return torch.mean(loss)

    def get_loss(self, batch_dict, device):
        nobs = batch_dict['obs_cond'].to(device).float().flatten(1)
        naction = batch_dict['expert_act'].to(device).float()


        if 'vla_act' in batch_dict:
            prior_action = batch_dict['vla_act'].to(device).float()
        else:
            prior_action = torch.randn(naction.shape).float().to(naction.device)

        obs_cond = nobs

        target = naction
        source = prior_action

        # ===== compute loss =====
        step = torch.rand(target.shape[0]).to(device)

        xt, noise = self.q_sample(step, source, target)

        v_loss = self.velocity_loss(v_net=self.net.v_net, t=step, xt=xt, x0=source, x1=target, cond=obs_cond)
        s_loss = self.score_loss(s_net=self.net.s_net, t=step, x_t=xt, z=noise, cond=obs_cond)
        b_loss = self.b_loss(b_net=self.net.b_net, t=step, xt=xt, x0=source, x1=target, z=noise, cond=obs_cond)

        loss = v_loss + s_loss + b_loss
        loss_info = {'v_loss': v_loss, 's_loss': s_loss, 'b_loss': b_loss}
        return loss, loss_info

    def q_sample(self, t, x0, x1):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """
        batch, *xdim = x0.shape
        t_batch = unsqueeze_xdim(t, xdim)
        t_batch = torch.clip(t_batch, self.t_min, 1.0 - self.t_min)

        gamma = self.gamma(t_batch)
        xt, z = self.interpolant(x0=x0, x1=x1, gamma=gamma, t=t_batch)

        return xt.detach(), z

    def sample(self, x_prior, cond, diffuse_step=10, recod_traj=False):
        """

        :param x_prior: (batch, feature)
        :param sde_type: vs, bs
        :param save_path:
        :return:
        """
        with self.ema.average_parameters():
            if self.sde_type == 'vs':
                x_target, x_target_traj = self.sde_vs(v_net=self.net.v_net, s_net=self.net.s_net, x_initial=x_prior, cond=cond,
                                                      delta_t=float(1.0 / diffuse_step), score_weight=1.0, direction='forward')
            elif self.sde_type == 'bs':
                x_target, x_target_traj = self.sde_bs(b_net=self.net.b_net, s_net=self.net.s_net, x_initial=x_prior, cond=cond,
                                                      delta_t=float(1.0 / diffuse_step), score_weight=1.0, direction='forward')
            else:
                raise NotImplementedError
        if recod_traj:
            return x_target, x_target_traj
        else:
            return x_target

    def sde_bs(self, b_net, s_net, x_initial, cond, delta_t=0.025, score_weight=1.0, direction='forward'):
        # Number of steps and samples
        n_steps = int(1.0 / delta_t)
        n_samples = x_initial.shape[0]

        # Create a tensor to hold the samples at each time step
        x_values = [[]] * (n_steps + 1)
        x_values[0] = x_initial

        # Simulate the SDE
        for t in range(1, n_steps + 1):
            current_x = x_values[t - 1]

            # Create a tensor of shape (n_samples, 1) filled with the current time value
            t_tensor = torch.full((n_samples,), t / n_steps).float().to(x_initial.device)
            t_tensor = torch.clip(t_tensor, self.t_min, 1.0 - self.t_min)

            if direction == 'forward':
                b_value = b_net(current_x, t_tensor, global_cond=cond).detach()

                s_value = s_net(current_x, t_tensor, global_cond=cond).detach()
                gamma_inv = self.gamma_inv(t_tensor)
            elif direction == 'backward':
                b_value = b_net(current_x, 1.0 - t_tensor, global_cond=cond).detach()

                s_value = s_net(current_x, 1.0 - t_tensor, global_cond=cond).detach()
                gamma_inv = self.gamma_inv(1.0 - t_tensor)
            else:
                raise NotImplementedError

            batch, *xdim = s_value.shape
            gamma_inv = unsqueeze_xdim(gamma_inv, xdim)
            s_value = s_value * gamma_inv

            # Generate the Wiener process increment
            dW = self.d * torch.randn_like(current_x).float().to(x_initial.device)

            if direction == 'forward':
                noise_scale = delta_t * torch.sqrt(2 * self.epsilon(t_tensor[0]))
                score_epsilon = score_weight * self.epsilon(t_tensor[0])
                new_x = current_x + (b_value + score_epsilon * s_value) * delta_t

            elif direction == 'backward':
                noise_scale = delta_t * torch.sqrt(2 * self.epsilon(1.0 - t_tensor[0]))
                score_epsilon = score_weight * self.epsilon(1.0 - t_tensor[0])
                new_x = current_x - (b_value - score_epsilon * s_value) * delta_t
            else:
                raise NotImplementedError
            new_x += noise_scale * dW
            x_values[t] = new_x

        return x_values[-1], x_values

    def sde_vs(self, v_net, s_net, x_initial, cond, delta_t=0.025, score_weight=1.0, direction='forward'):
        n_steps = int(1.0 / delta_t)
        n_samples = x_initial.shape[0]

        # Create a tensor to hold the samples at each time step
        x_values = [[]] * (n_steps + 1)
        x_values[0] = x_initial

        # Simulate the SDE
        for t in range(1, n_steps + 1):
            current_x = x_values[t - 1]

            # Create a tensor of shape (n_samples, 1) filled with the current time value
            t_tensor = torch.full((n_samples,), t / n_steps).float().to(x_initial.device)
            t_tensor = torch.clip(t_tensor, self.t_min, 1.0 - self.t_min)

            if direction == 'forward':
                gamma_t, dot_gamma_t = self.gamma(t_tensor), self.gamma_der(t_tensor)
                v_value = v_net(current_x, t_tensor, global_cond=cond).detach()
                s_value = s_net(current_x, t_tensor, global_cond=cond).detach()
                gamma_inv = self.gamma_inv(t_tensor)
            elif direction == 'backward':
                gamma_t, dot_gamma_t = self.gamma(1.0 - t_tensor), self.gamma_der(1.0 - t_tensor)
                v_value = v_net(current_x, 1.0 - t_tensor, global_cond=cond).detach()
                s_value = s_net(current_x, 1.0 - t_tensor, global_cond=cond).detach()
                gamma_inv = self.gamma_inv(1.0 - t_tensor)
            else:
                raise NotImplementedError

            batch, *xdim = s_value.shape
            gamma_inv = unsqueeze_xdim(gamma_inv, xdim)
            s_value = s_value * gamma_inv

            dot_gamma_gamma_t = dot_gamma_t.float().to(x_initial.device) * gamma_t.float().to(x_initial.device)
            dot_gamma_gamma_t = unsqueeze_xdim(dot_gamma_gamma_t, xdim)
            b_value = v_value - dot_gamma_gamma_t * s_value * self.epsilon(t_tensor[0])

            # Generate the Wiener process increment
            dW = self.d * torch.randn_like(current_x).float().to(x_initial.device)

            if direction == 'forward':
                noise_scale = delta_t * torch.sqrt(2 * self.epsilon(t_tensor[0]))
                score_epsilon = score_weight * self.epsilon(t_tensor[0])
                new_x = current_x + (b_value + score_epsilon * s_value) * delta_t
            elif direction == 'backward':
                noise_scale = delta_t * torch.sqrt(2 * self.epsilon(1.0 - t_tensor[0]))
                score_epsilon = score_weight * self.epsilon(1.0 - t_tensor[0])
                new_x = current_x - (b_value - score_epsilon * s_value) * delta_t
            else:
                raise NotImplementedError
            new_x += noise_scale * dW
            x_values[t] = new_x

        return x_values[-1], x_values

    def train(self):
        """
        Set the model to training mode.
        This affects batch normalization, dropout, and similar layers.
        """
        if self.net is not None:
            self.net.train()
            # Specifically ensure each network component is in train mode
            if hasattr(self.net, 'v_net'):
                self.net.v_net.train()
            if hasattr(self.net, 's_net'):
                self.net.s_net.train()
            if hasattr(self.net, 'b_net'):
                self.net.b_net.train()
        return self

    def eval(self):
        """
        Set the model to evaluation mode.
        This affects batch normalization, dropout, and similar layers.
        """
        if self.net is not None:
            self.net.eval()
            # Specifically ensure each network component is in eval mode
            if hasattr(self.net, 'v_net'):
                self.net.v_net.eval()
            if hasattr(self.net, 's_net'):
                self.net.s_net.eval()
            if hasattr(self.net, 'b_net'):
                self.net.b_net.eval()
        return self

    def load_model(self, model_args, device):

        self.load_model_args(model_args)

        if model_args['net_type'] == 'unet1D_si':
            self.net = InterpolantsConditionalUnet1D(
                input_dim=model_args['action_dim'],
                global_cond_dim=model_args['obs_dim'] * model_args['obs_horizon']
            )
        else:
            raise NotImplementedError

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.75)

        if model_args['pretrain']:
            checkpoint = torch.load(os.path.join(model_args['ckpt_path'], "bridge_model.pt"), map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            self.ema.load_state_dict(checkpoint["ema"])

        self.net.to(device)
        self.ema.to(device)

    def save_model(self, ckpt_path):
        torch.save({
            "net": self.net.state_dict(),
            "ema": self.ema.state_dict(),
        }, os.path.join(ckpt_path, "bridge_model.pt"))