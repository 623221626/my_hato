import collections
import copy
import os

import numpy as np
import torch
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from models import *
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / ((stats["max"] - stats["min"]) + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"] + 1e-8) + stats["min"]
    return data


class DiffusionPolicy:
    def __init__(
        self,
        obs_horizon,#模型在预测动作时考虑的历史观测帧数，默认为1
        obs_dim,# = 所有编码器输出特征的总维度 = eef_dim + hand_pos_dim + image_dim + pos_dim + touch_dim
        pred_horizon,#模型一次预测的动作步数
        action_horizon,#每轮推理时实际执行的动作步数，影响在eval函数中控制每次从预测结果中取多少步动作
        action_dim,
        representation_type,  # pos, img, touch, eef
        encoders,# 各模态编码器
        num_diffusion_iters=100,
        without_sampling=False,#是否跳过扩散过程，直接使用简单行为克隆（BC）模型，默认为false
        weight_decay=1e-6,#权重衰减率
        use_ddim=False,#是否使用确定性采样（DDIM）而非随机采样（DDPM）
        binarize_touch=False,#是否将触觉传感器数据二值化（如>阈值=1，否则=0）
        policy_dropout_rate=0.0,
    ):
        for rt in representation_type:
            assert rt in encoders, f"{rt} not in encoders"
        self.representation_type = representation_type
        self.encoders = encoders
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.data_stat = None
        self.writer = None
        self.without_sampling = without_sampling
        self.binarize_touch = binarize_touch

        if self.without_sampling:#默认为false
            bc_actor = SimpleBCModel(
                input_dim=obs_dim * obs_horizon,
                output_dim=self.action_dim * self.pred_horizon,
                dropout_rate=policy_dropout_rate,
            )
            # the final arch has 2 parts
            self.nets = nn.ModuleDict({"bc_actor": bc_actor})
        else:
            """
            Diffusion Model
            训练方式：预测扩散噪声，通过多步迭代生成动作。
            """
            noise_pred_net = ConditionalUnet1D(#?
                input_dim=action_dim, # 输入为噪声动作
                global_cond_dim=obs_dim * obs_horizon # 观测作为全局条件
            )
            # the final arch has 2 parts
            self.nets = nn.ModuleDict({"noise_pred_net": noise_pred_net}) # 存储模型

        for rt in representation_type:
            self.nets[f"{rt}_encoder"] = encoders[rt]

        self.num_diffusion_iters = num_diffusion_iters

        if use_ddim:#默认为false
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule="squaredcos_cap_v2",
                # clip output to [-1,1] to improve stability
                clip_sample=True,
                # our network predicts noise (instead of denoised action)
                prediction_type="epsilon",
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,#从真实数据到纯噪声需要多少步。
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule="squaredcos_cap_v2",#squaredcos：使用平方余弦函数定义噪声强度，后期降噪更平滑。cap_v2：对噪声强度进行裁剪优化（防止极端值）。
                # clip output to [-1,1] to improve stability
                clip_sample=True,#将生成的噪声限制在[-1,1]范围内。
                # our network predicts noise (instead of denoised action)
                prediction_type="epsilon",#模型预测的是噪声本身（epsilon），而不是去噪后的动作。
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Exponential Moving Average of the model weights
        self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)#来自 diffusers 库的工具，专门用来跟踪模型参数的移动平均。以后推理时使用平滑后的参数，模型输出更稳定，减少训练时参数波动带来的影响。
        self.ema_nets = copy.deepcopy(self.nets)#复制一份原始模型的参数（如卷积层、全连接层），用于存储 EMA 的平滑参数。

        # Standard ADAM optimizer
        self.optimizer = torch.optim.AdamW(#AdamW：Adam 的改进版。Adam：一种自适应学习率的优化算法，能自动调整每个参数的更新步长。
            params=self.nets.parameters(), lr=1e-4, weight_decay=weight_decay
        )

    def set_lr_scheduler(self, num_training_steps):
        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps,
        )

    def to(self, device):
        self.device = device
        self.nets.to(device)
        self.ema.to(device)
        self.ema_nets.to(device)

    def train(
        self,
        num_epochs,#训练总轮数
        dataloader,#训练数据加载器
        eval_data=None,
        save_path=None,
        save_freq=10,
        eval_freq=10,
        wandb_logger=None,
        eval=False,
    ):
        """
        遍历每个epoch，再在其中遍历每个batch：
        1. 获取batch数据，并更新obs_deque；
        2. 获取batch动作，并更新stats；
        3. 训练和评估模型（只有训练集参与训练，测试集eval_data只用来定期评估mse）
            在训练中，特征X是obs_cond，来自从nbatch中提取的多模态数据（比如eef、hand_pos等），
                    标签Y是动作序列nbatch["naction"]，形状为(B, pred_horizon, action_dim)
                    （训练阶段的）预测值是模型预测的噪声(noise_pred)而非直接预测动作序列。（推理阶段的）预测值是从纯噪声开始，通过逆扩散生成的动作序列pred_action
        4. 保存模型；
        5. 保存统计数据。
        """
        if eval:
            nets = self.ema_nets
            nets.eval()
            action_mse = []
        else:
            nets = self.nets
            nets.train()#仅用于切换模型到训练模式，而完整的训练流程（数据预处理、扩散过程、参数更新、输出等）均由后续代码实现

        if self.writer is None and save_path is not None:
            #TensorBoard 日志初始化
            # get the name from save_path
            name = os.path.basename(save_path)
            self.writer = SummaryWriter(os.path.join("./runs", name))

        with tqdm(range(num_epochs), desc="Epoch") as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                # batch loop
                epoch_loss = list()
                with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:#nbatch包含了action及eef,hand_pos等6个成员张量
                        # data normalized in dataset
                        # device transfer
                        naction = nbatch["action"].to(self.device)#从批次中提取动作数据并移动到指定设备（CPU/GPU），形状：(B, pred_horizon, action_dim)
                        B = naction.shape[0]
                        features = []

                        ### IMPT: make sure input is always in this order
                        # eef, hand_pos, img, pos, touch
                        for data_key in [
                            dk
                            for dk in ["eef", "hand_pos", "img", "pos", "touch"]
                            if dk in self.representation_type
                        ]:
                            nsample = nbatch[data_key][:, : self.obs_horizon].to(
                                self.device
                            )
                            if data_key == "img":
                                #将图像按时间步拆分，分别通过对应的编码器，最后拼接到features。
                                images = [
                                    nsample[:, :, i] for i in range(nsample.shape[2])
                                ]  # [B, obs_horizon, M, C, H, W]
                                image_features = [
                                    nets[f"{data_key}_encoder"][i](
                                        image.flatten(end_dim=1)
                                    )
                                    for i, image in enumerate(images)
                                ]
                                image_features = torch.stack(image_features, dim=2)
                                image_features = image_features.reshape(
                                    *nsample.shape[:2], -1
                                )
                                features.append(image_features)
                            else:
                                #其他模态直接通过编码器处理，输出特征展平后拼接到 features。
                                nfeat = nets[f"{data_key}_encoder"](
                                    nsample.flatten(end_dim=1)
                                )
                                nfeat = nfeat.reshape(*nsample.shape[:2], -1)
                                features.append(nfeat)
                        """
                        #将多模态特征在特征维度（最后一维）上拼接。
                        输入：
                            features 是一个列表，包含各模态编码后的特征张量。
                            每个元素形状为 (B, obs_horizon, feature_dim)，其中：
                            B 是批次大小（batch size）
                            obs_horizon 是历史观测帧数
                            feature_dim 是该模态的编码特征维度
                        输出：
                            obs_features 形状为 (B, obs_horizon, total_feature_dim)
                            total_feature_dim = 所有模态特征维度之和（即 obs_dim）
                        示例： 假设：
                            features 包含两个模态特征：eef_feat (dim=64) 和 img_feat (dim=128)
                            每个特征形状为 (32, 2, 64) 和 (32, 2, 128)（B=32, obs_horizon=2）
                            则输出 obs_features.shape = (32, 2, 192)（64+128=192）
                        """
                        obs_features = torch.cat(features, dim=-1)
                        """
                        将时间步和特征维度合并，生成全局条件向量。
                        输入：
                            obs_features 形状为 (B, obs_horizon, total_feature_dim)
                        输出：
                            obs_cond 形状为 (B, obs_horizon * total_feature_dim)
                        """
                        # (B, obs_horizon * obs_dim)
                        obs_cond = obs_features.flatten(start_dim=1)

                        if self.without_sampling:
                            action = nets["bc_actor"](obs_cond)
                            action = action.reshape(
                                -1, self.pred_horizon, self.action_dim
                            )
                            # L2 loss
                            loss = nn.functional.mse_loss(action, naction)
                        else:
                            """
                            真实动作 → 添加噪声 → 输入模型 → 预测噪声 → 计算损失 → 反向传播
                            """
                            # sample noise to add to actions
                            noise = torch.randn(naction.shape, device=self.device)

                            # Training
                            if not eval:
                                """
                                随机采样噪声和时间步，添加噪声noise到动作naction上，形成noisy_actions。
                                使用 noise_pred_net根据noisy_actions和观测条件来预测噪声，计算 MSE 损失。
                                """
                                # sample a diffusion iteration for each data point
                                timesteps = torch.randint(#  生成B个随机数，范围是[0, num_train_timesteps)
                                    0,
                                    self.noise_scheduler.config.num_train_timesteps,
                                    (B,),
                                    device=self.device,
                                ).long()

                                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                                # (this is the forward diffusion process)
                                noisy_actions = self.noise_scheduler.add_noise(
                                    naction, noise, timesteps
                                )
                                # predict the noise residual
                                noise_pred = nets["noise_pred_net"](
                                    noisy_actions, timesteps, global_cond=obs_cond
                                )

                                # L2 loss
                                loss = nn.functional.mse_loss(noise_pred, noise)

                            # Evaluation
                            else:
                                """
                                从纯噪声开始，通过逆扩散过程生成动作。
                                计算归一化和非归一化的 MSE 损失。
                                """
                                noisy_action = noise
                                pred_action = noisy_action

                                self.noise_scheduler.set_timesteps(
                                    self.num_diffusion_iters
                                )

                                for k in self.noise_scheduler.timesteps:
                                    # predict noise
                                    noise_pred = nets["noise_pred_net"](
                                        sample=pred_action,
                                        timestep=k,
                                        global_cond=obs_cond,
                                    )

                                    # inverse diffusion step (remove noise)
                                    pred_action = self.noise_scheduler.step(
                                        model_output=noise_pred,
                                        timestep=k,
                                        sample=pred_action,
                                    ).prev_sample

                                loss = nn.functional.mse_loss(naction, pred_action)

                                unnormalized_naction = unnormalize_data(
                                    naction.detach().cpu().numpy(),
                                    self.data_stat["action"],
                                )
                                unnormalized_pred_action = unnormalize_data(
                                    pred_action.detach().cpu().numpy(),
                                    self.data_stat["action"],
                                )
                                unnormalized_loss = nn.functional.mse_loss(
                                    torch.tensor(unnormalized_naction),
                                    torch.tensor(unnormalized_pred_action),
                                )

                        if not eval:
                            # optimize
                            loss.backward()#反向传播
                            #优化器更新
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # step lr scheduler every batch
                            # this is different from standard pytorch behavior
                            self.lr_scheduler.step()#学习率调度

                            # update Exponential Moving Average of the model weights
                            self.ema.step(nets.parameters())#EMA 更新。
                        else:
                            action_mse.append(unnormalized_loss.item())

                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))
                if self.writer is not None:
                    self.writer.add_scalar("Loss", np.mean(epoch_loss), epoch_idx)

                if eval:
                    return np.mean(epoch_loss), np.mean(action_mse)

                if wandb_logger is not None:
                    wandb_logger.step()
                    wandb_logger.log({"Loss": np.mean(epoch_loss), "epoch": epoch_idx})
                if (
                    save_path is not None
                    and epoch_idx % save_freq == 0
                    and epoch_idx != 0
                ):
                    model_path = os.path.join(
                        save_path, f"model_epoch_{epoch_idx}.ckpt"
                    )
                    self.save(model_path)

                # save last checkpoint
                model_path = os.path.join(save_path, f"last.ckpt")
                self.save(model_path)

                #每隔 eval_freq 轮调用 self.eval() 进行验证，记录动作 MSE 和归一化 MSE。
                if eval_data is not None and epoch_idx % eval_freq == 0:
                    self.to_ema()
                    self.ema_nets.eval()
                    print("Evaluating one trajectory...")
                    obs, action = eval_data
                    _, mse, normalized_mse = self.eval(obs, action)
                    self.writer.add_scalar("Action_MSE", mse, epoch_idx)
                    self.writer.add_scalar("Normalized_MSE", normalized_mse, epoch_idx)

                    if wandb_logger is not None:
                        wandb_logger.log({"Action_MSE": mse})
                        wandb_logger.log({"Normalized_MSE": normalized_mse})
                    print(f"Action_MSE: {mse}, Normalized_MSE: {normalized_mse}")
                    self.ema_nets.train()

    def eval(self, obs, action):
        obs_deque = collections.deque(
            [obs[0]] * self.obs_horizon, maxlen=self.obs_horizon
        )
        actions_pred = []

        i = 0
        while i < len(obs) - self.action_horizon:
            action_pred = self.forward(self.data_stat, obs_deque)
            for j in range(self.action_horizon):
                actions_pred.append(action_pred[j])
                obs_deque.append(obs[i + j])
            i += self.action_horizon

        actions_pred = np.array(actions_pred)
        action = np.array(action)
        mse = mse_loss(
            torch.tensor(actions_pred), torch.tensor(action[: len(actions_pred)])
        )

        normalized_action = normalize_data(action, self.data_stat["action"])
        normalized_action_pred = normalize_data(actions_pred, self.data_stat["action"])

        normalized_mse = mse_loss(
            torch.tensor(normalized_action_pred),
            torch.tensor(normalized_action[: len(actions_pred)]),
        )

        return actions_pred, mse, normalized_mse

    def to_ema(self):
        # Weights of the EMA model
        # is used for inference
        self.ema.copy_to(self.ema_nets.parameters())

    def load(self, path):
        def rename_key(old_key):
            new_key = old_key.replace("image", "img")
            unexpected = [
                "pos_encoder.encoder.mlp.0.weight",
                "pos_encoder.encoder.mlp.0.bias",
                "pos_encoder.encoder.mlp.2.weight",
                "pos_encoder.encoder.mlp.2.bias",
                "touch_encoder.encoder.mlp.0.weight",
                "touch_encoder.encoder.mlp.0.bias",
                "touch_encoder.encoder.mlp.2.weight",
                "touch_encoder.encoder.mlp.2.bias",
            ]
            missing = [
                "pos_encoder.linear.mlp.0.weight",
                "pos_encoder.linear.mlp.0.bias",
                "pos_encoder.linear.mlp.2.weight",
                "pos_encoder.linear.mlp.2.bias",
                "touch_encoder.linear.mlp.0.weight",
                "touch_encoder.linear.mlp.0.bias",
                "touch_encoder.linear.mlp.2.weight",
                "touch_encoder.linear.mlp.2.bias",
            ]
            for i, u in enumerate(unexpected):
                new_key = new_key.replace(u, missing[i])
            return new_key

        basename = os.path.basename(path)
        dirname = os.path.dirname(path)

        state_dict = torch.load(path, map_location="cuda")
        # rename model keys for backward compatibility
        state_dict = {rename_key(k): v for k, v in state_dict.items()}

        self.nets.load_state_dict(state_dict)

        if os.path.exists(os.path.join(dirname, "ema_" + basename)):
            ema_state_dict = torch.load(
                os.path.join(dirname, "ema_" + basename), map_location="cuda"
            )
            ema_state_dict = {rename_key(k): v for k, v in ema_state_dict.items()}
            self.ema_nets.load_state_dict(ema_state_dict)
        else:
            self.ema_nets.load_state_dict(state_dict)

    def save(self, path):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.to_ema()
        torch.save(self.ema_nets.state_dict(), os.path.join(dirname, "ema_" + basename))
        torch.save(self.nets.state_dict(), path)

    def _get_data_forward(self, stats, obs_deque, data_key):
        sample = np.stack([x[data_key] for x in obs_deque])
        if data_key != "img" and (data_key != "touch" or not self.binarize_touch):
            # image is already normalized
            sample = normalize_data(sample, stats=stats[data_key])
        sample = (
            torch.from_numpy(sample).to(self.device, dtype=torch.float32).unsqueeze(0)
        )
        return sample

    def forward(self, stats, obs_deque, num_diffusion_iters=None):
        self.ema_nets.eval()

        if not num_diffusion_iters:
            num_diffusion_iters = self.num_diffusion_iters

        with torch.no_grad():
            features = []

            ### IMPT: make sure input is always in this order
            # eef, hand_pos, img, pos, touch
            for data_key in [
                dk
                for dk in ["eef", "hand_pos", "img", "pos", "touch"]
                if dk in self.representation_type
            ]:
                sample = self._get_data_forward(stats, obs_deque, data_key)
                if data_key == "img":
                    images = [
                        sample[:, :, i] for i in range(sample.shape[2])
                    ]  # [1, obs_horizon, M, C, H, W]
                    image_features = [
                        self.ema_nets[f"{data_key}_encoder"][i](
                            image.flatten(end_dim=1)
                        )
                        for i, image in enumerate(images)
                    ]
                    image_features = torch.stack(image_features, dim=2)
                    image_features = image_features.reshape(*sample.shape[:2], -1)
                    features.append(image_features)
                else:
                    feat = self.ema_nets[f"{data_key}_encoder"](
                        sample.flatten(end_dim=1)
                    )
                    feat = feat.reshape(*sample.shape[:2], -1)
                    features.append(feat)

            obs_features = torch.cat(features, dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

            if self.without_sampling:
                action = self.ema_nets["bc_actor"](obs_cond)
                naction = action.reshape(-1, self.pred_horizon, self.action_dim)
            else:
                noisy_action = torch.randn(
                    (1, self.pred_horizon, self.action_dim), device=self.device
                )
                naction = noisy_action

                self.noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in self.noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = self.ema_nets["noise_pred_net"](
                        sample=naction, timestep=k, global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample

        # unnormalize action
        naction = naction.detach().to("cpu").numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats["action"])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]

        return action

    def eval_loader(self, eval_loader):
        self.ema_nets.eval()
        mse = self.train(num_epochs=1, dataloader=eval_loader, eval=True)
        return mse
