#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.simplefilter('default')

import paddle.fluid as fluid
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers
from vtrace import from_importance_weights
# from parl.layers import Normal
from paddle.fluid.layers import Normal
from parl.core.fluid.policy_distribution import CategoricalDistribution
from parl.core.fluid.plutils import inverse

__all__ = ['DVtrace']
epsilon = 1e-6

class VTraceLoss(object):
    def __init__(self,
                 behaviour_actions_log_probs,
                 target_actions_log_probs,
                 policy_entropy,
                 dones,
                 discount,
                 rewards,
                 values,
                 bootstrap_value,
                 entropy_coeff=-0.01,
                 vf_loss_coeff=0.5,
                 clip_rho_threshold=1.0,
                 clip_pg_rho_threshold=1.0):
        """Policy gradient loss with vtrace importance weighting.

        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
        batch_size. The reason we need to know `B` is for V-trace to properly
        handle episode cut boundaries.

        Args:
            behaviour_actions_log_probs: A float32 tensor of shape [T, B].
            target_actions_log_probs: A float32 tensor of shape [T, B].
            policy_entropy: A float32 tensor of shape [T, B].
            dones: A float32 tensor of shape [T, B].
            discount: A float32 scalar.
            rewards: A float32 tensor of shape [T, B].
            values: A float32 tensor of shape [T, B].
            bootstrap_value: A float32 tensor of shape [B].
        """

        self.vtrace_returns = from_importance_weights(
            behaviour_actions_log_probs=behaviour_actions_log_probs,
            target_actions_log_probs=target_actions_log_probs,
            discounts=inverse(dones) * discount,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold)

        # The policy gradients loss
        self.pi_loss = -1.0 * layers.reduce_sum(
            target_actions_log_probs * self.vtrace_returns.pg_advantages)

        # The baseline loss
        delta = values - self.vtrace_returns.vs
        self.vf_loss = 0.5 * layers.reduce_sum(layers.square(delta))

        # The entropy loss (We want to maximize entropy, so entropy_ceoff < 0)
        self.entropy = layers.reduce_sum(policy_entropy)

        # The summed weighted loss
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff +
                           self.entropy * entropy_coeff)


class DVtrace(Algorithm):
    def __init__(self,
                 model,
                 max_action,
                 hyperparas=None,
                 sample_batch_steps=None,
                 gamma=None,
                 vf_loss_coeff=None,
                 clip_rho_threshold=None,
                 clip_pg_rho_threshold=None):
        """ IMPALA algorithm
        
        Args:
            model (parl.Model): forward network of policy and value
            hyperparas (dict): (deprecated) dict of hyper parameters.
            sample_batch_steps (int): steps of each environment sampling.
            gamma (float): discounted factor for reward computation.
            vf_loss_coeff (float): coefficient of the value function loss.
            clip_rho_threshold (float): clipping threshold for importance weights (rho).
            clip_pg_rho_threshold (float): clipping threshold on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
        """
        if hyperparas is not None:
            warnings.warn(
                "the `hyperparas` argument of `__init__` function in `parl.Algorithms.IMPALA` is deprecated since version 1.2 and will be removed in version 1.3.",
                DeprecationWarning,
                stacklevel=2)
            self.sample_batch_steps = hyperparas['sample_batch_steps']
            self.gamma = hyperparas['gamma']
            self.vf_loss_coeff = hyperparas['vf_loss_coeff']
            self.clip_rho_threshold = hyperparas['clip_rho_threshold']
            self.clip_pg_rho_threshold = hyperparas['clip_pg_rho_threshold']
        else:
            assert isinstance(sample_batch_steps, int)
            assert isinstance(gamma, float)
            assert isinstance(vf_loss_coeff, float)
            assert isinstance(clip_rho_threshold, float)
            assert isinstance(clip_pg_rho_threshold, float)
            self.sample_batch_steps = sample_batch_steps
            self.gamma = gamma
            self.vf_loss_coeff = vf_loss_coeff
            self.clip_rho_threshold = clip_rho_threshold
            self.clip_pg_rho_threshold = clip_pg_rho_threshold

        self.model = model
        self.max_action = max_action

    def learn(self, obs, actions, means, log_std, rewards, dones,
              learning_rate, entropy_coeff):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
            actions: An int64 tensor of shape [B].
            behaviour_logits: A float32 tensor of shape [B, NUM_ACTIONS].
            rewards: A float32 tensor of shape [B].
            dones: A float32 tensor of shape [B].
            learning_rate: float scalar of learning rate.
            entropy_coeff: float scalar of entropy coefficient.
        """
        values = self.model.value(obs)
        # pi
        log_std = layers.exp(log_std)
        normal_pi = Normal(means, log_std)
        # x_t1 = normal_pi.sample([1])[0]
        # x_t1.stop_gradient = True
        y_t1 = actions / self.max_action
        # action1 = y_t1 * self.max_action
        log_prob1 = normal_pi.log_prob(actions)
        log_prob1 -= layers.log(self.max_action * (1 - layers.pow(y_t1, 2)) + epsilon)
        log_prob1 = layers.reduce_sum(log_prob1, dim=1, keep_dim=True)
        log_prob_pi = layers.squeeze(log_prob1, axes=[1])

        # mu
        actions_mu, log_std_mu = self.model.policy(obs)
        log_std_mu = layers.exp(log_std_mu)
        normal_mu = Normal(actions_mu, log_std_mu)
        # x_t2 = normal_mu.sample([1])[0]
        # x_t2.stop_gradient = True
        # y_t2 = actions
        # action2 = y_t2 * self.max_action
        log_prob2 = normal_mu.log_prob(actions)
        log_prob2 -= layers.log(self.max_action * (1 - layers.pow(y_t1, 2)) + epsilon)
        log_prob2 = layers.reduce_sum(log_prob2, dim=1, keep_dim=True)
        log_prob_mu = layers.squeeze(log_prob2, axes=[1])

        # target_policy_distribution = CategoricalDistribution(target_logits)
        # behaviour_policy_distribution = CategoricalDistribution(
        #     behaviour_logits)

        policy_entropy = normal_mu.entropy()
        # policy_entropy = layers.reduce_mean(policy_entropy, dim=1)
        target_actions_log_probs = log_prob_mu
        behaviour_actions_log_probs = log_prob_pi

        # Calculating kl for debug
        # kl = target_policy_distribution.kl(behaviour_policy_distribution)
        kl = normal_mu.kl_divergence(normal_pi)
        kl = layers.reduce_mean(kl, dim=1)
        # kl = layers.unsqueeze(kl, axes=[1])
        """
        Split the tensor into batches at known episode cut boundaries. 
        [B * T] -> [T, B]
        """
        T = self.sample_batch_steps

        def split_batches(tensor):
            B = tensor.shape[0] // T
            splited_tensor = layers.reshape(tensor,
                                            [B, T] + list(tensor.shape[1:]))
            # transpose B and T
            return layers.transpose(
                splited_tensor, [1, 0] + list(range(2, 1 + len(tensor.shape))))

        behaviour_actions_log_probs = split_batches( behaviour_actions_log_probs)
        target_actions_log_probs = split_batches(target_actions_log_probs)
        policy_entropy = split_batches(policy_entropy)
        dones = split_batches(dones)
        rewards = split_batches(rewards)
        values = split_batches(values)

        # [T, B] -> [T - 1, B] for V-trace calc.
        behaviour_actions_log_probs = layers.slice(behaviour_actions_log_probs, axes=[0], starts=[0], ends=[-1])
        target_actions_log_probs = layers.slice(
            target_actions_log_probs, axes=[0], starts=[0], ends=[-1])
        policy_entropy = layers.slice(
            policy_entropy, axes=[0], starts=[0], ends=[-1])
        dones = layers.slice(dones, axes=[0], starts=[0], ends=[-1])
        rewards = layers.slice(rewards, axes=[0], starts=[0], ends=[-1])
        bootstrap_value = layers.slice(
            values, axes=[0], starts=[T - 1], ends=[T])
        values = layers.slice(values, axes=[0], starts=[0], ends=[-1])

        bootstrap_value = layers.squeeze(bootstrap_value, axes=[0])

        vtrace_loss = VTraceLoss(
            behaviour_actions_log_probs=behaviour_actions_log_probs,
            target_actions_log_probs=target_actions_log_probs,
            policy_entropy=policy_entropy,
            dones=dones,
            discount=self.gamma,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=self.vf_loss_coeff,
            clip_rho_threshold=self.clip_rho_threshold,
            clip_pg_rho_threshold=self.clip_pg_rho_threshold)

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=40.0))

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate)
        optimizer.minimize(vtrace_loss.total_loss)
        return vtrace_loss, kl

    def sample(self, obs):
        mean, log_std = self.model.policy(obs)
        std = layers.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.sample([1])[0]
        y_t = layers.tanh(x_t)
        action = y_t * self.max_action
        # log_prob = normal.log_prob(x_t)
        # log_prob -= layers.log(self.max_action * (1 - layers.pow(y_t, 2)) + epsilon)
        # log_prob = layers.reduce_sum(log_prob, dim=1, keep_dim=True)
        # log_prob = layers.squeeze(log_prob, axes=[1])
        return action, mean, log_std

    # def sample(self, obs):
    #     """
    #     Args:
    #         obs: An float32 tensor of shape ([B] + observation_space).
    #              E.g. [B, C, H, W] in atari.
    #     """
    #     logits = self.model.policy(obs)
    #     policy_dist = CategoricalDistribution(logits)
    #     sample_actions = policy_dist.sample()
    #     return sample_actions, logits

    # def predict(self, obs):
    #     """
    #     Args:
    #         obs: An float32 tensor of shape ([B] + observation_space).
    #              E.g. [B, C, H, W] in atari.
    #     """
    #     logits = self.model.policy(obs)
    #     probs = layers.softmax(logits)

    #     predict_actions = layers.argmax(probs, 1)

    #     return predict_actions
