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

import paddle.fluid as fluid
import parl
from parl import layers

LOG_SIG_MAX = 1.0
LOG_SIG_MIN = 0.0


class MujocoModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 400
        hid2_size = 300
        # self.fc01 = layers.fc(size=hid1_size, act='relu')
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.mean_linear = layers.fc(size=act_dim, act='tanh')
        self.log_std_linear = layers.fc(size=act_dim, act='tanh')

        # self.fc4 = layers.fc(size=hid1_size, act='relu')
        # self.fc5 = layers.fc(size=hid2_size, act='relu')
        self.value_fc = layers.fc(size=1, act=None)

    def policy(self, obs):
        # fc01 = self.fc01(obs)
        obs = layers.flatten(obs, axis=1)
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        means = self.mean_linear(hid2)
        log_std = self.log_std_linear(hid2)
        log_std = layers.clip(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return means, log_std

    def value(self, obs):
        obs = layers.flatten(obs, axis=1)
        hid1 = self.fc1(obs)
        # concat1 = layers.concat([hid1, act], axis=1)
        hid2 = self.fc2(hid1)
        V = self.value_fc(hid2)
        V = layers.squeeze(V, axes=[1])
        return V

# class CriticModel(parl.Model):
#     def __init__(self):
#         hid1_size = 400
#         hid2_size = 300

# self.fc1 = layers.fc(size=hid1_size, act='relu')
# self.fc2 = layers.fc(size=hid2_size, act='relu')
# self.fc3 = layers.fc(size=1, act=None)

#         self.fc4 = layers.fc(size=hid1_size, act='relu')
#         self.fc5 = layers.fc(size=hid2_size, act='relu')
#         self.fc6 = layers.fc(size=1, act=None)

#     def value(self, obs, act):
#         hid1 = self.fc1(obs)
#         concat1 = layers.concat([hid1, act], axis=1)
#         Q1 = self.fc2(concat1)
#         Q1 = self.fc3(Q1)
#         Q1 = layers.squeeze(Q1, axes=[1])
