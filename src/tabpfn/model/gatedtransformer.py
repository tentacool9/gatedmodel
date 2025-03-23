from typing import Any, Mapping

import torch.nn as nn
import torch
from functorch import einops

from tabpfn.model.encoders import SequentialEncoder
from tabpfn.model.transformer import PerFeatureTransformer


class LearnedSampleGating(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], a=1.0, sigma=0.5):
        """
        Args:
            input_dim (int): The dimension of the input feature vector for each sample.
            hidden_dims (list of int): Hidden dimensions for the gating network.
            a (float): Scaling parameter in the hard sigmoid.
            sigma (float): Standard deviation for the Gaussian noise.
        """
        super(LearnedSampleGating, self).__init__()
        self.a = a
        self.sigma = sigma

        # Build the gating network: it processes each sample's feature vector and outputs a scalar.
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Fully connected layers to compute an intermediate representation.
        self.fc = nn.Sequential(*layers)
        # Final layer to produce one raw gate value (alpha) per sample.
        self.gate_fc = nn.Linear(prev_dim, 1)

    def hard_sigmoid(self, x):
        """
        Hard sigmoid: a piecewise linear approximation of the sigmoid function.
        Computes: f(x) = clip(a * x + 0.5, 0, 1)
        """
        return torch.clamp(self.a * x + 0.5, min=0.0, max=1.0)

    def forward(self, x, train_gates=True):
        """
        Forward pass of the sample gating mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (n, 1, dim) where n is the number of samples.
            train_gates (bool): If True, add noise for stochastic gating (used during training).

        Returns:
            x_gated (torch.Tensor): The gated sample feature vectors of shape (n, 1, dim) (each sample is scaled).
            gate (torch.Tensor): The gating values (in [0,1]) for each sample, shape (n, 1).
            raw_gate (torch.Tensor): The raw output (alpha) before noise and activation, shape (n, 1).
            weighted_sum (torch.Tensor): A combined representation computed as the weighted sum of sample features,
                                         following the operation: row vector (1 x n) * Identity (n x n) * X,
                                         yielding shape (1, dim).
        """
        # (batch ,seq_len, features_per_group, embedding)

        batch , seq, feat_p_g, embed = x.size()
        x_flat = x.mean(dim=2)

        # Compute an intermediate representation and then a raw gate value per sample.
        hidden = self.fc(x_flat)  # shape: (batch,seq, hidden_dim)
        raw_gate = self.gate_fc(hidden)  # shape: (batch, seq, 1)

        # Optionally add Gaussian noise to encourage exploration during training.
        if self.training or train_gates:
            noise = torch.randn_like(raw_gate)
            raw_gate = raw_gate + self.sigma * noise

        # Pass through the hard sigmoid to obtain a gating value between 0 and 1.
        gate = self.hard_sigmoid(raw_gate)  # shape: (batch, seq , 1)

        # Apply the sample gate to each sample's feature vector.
        # This is equivalent to forming a diagonal matrix from the gate vector and multiplying:
        # x_gated = diag(gate) * x_flat.
        x_gated = gate * x_flat  # shape: (batch, seq, dim)

        # Reshape x_gated back to (n, 1, dim) if needed.
        x_gated = x_gated.unsqueeze(2).expand(-1, -1, feat_p_g, -1)

        return x_gated, gate, raw_gate


class LearnedFeatureGating(nn.Module):
    def __init__(self, input_dim, gating_hidden_dims=[64, 32], a=1.0, sigma=0.5):
        """
        Args:
            input_dim (int): The dimension of the input features.
            gating_hidden_dims (list of int): Hidden dimensions for the gating network.
            a (float): Scaling parameter in the hard sigmoid.
            sigma (float): Standard deviation for the Gaussian noise.
        """
        super(LearnedFeatureGating, self).__init__()
        self.a = a
        self.sigma = sigma

        # Build the gating network: a series of fully connected layers
        layers = []
        prev_dim = input_dim
        for h_dim in gating_hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())  # Activation; can be modified as needed.
            prev_dim = h_dim

        # The gating network's hidden layers
        self.gating_layers = nn.Sequential(*layers)
        # Final layer to produce raw gate values (alpha) for each feature
        self.alpha_layer = nn.Linear(prev_dim, input_dim)

    def hard_sigmoid(self, x):
        """
        Hard sigmoid: a piecewise linear approximation of the sigmoid function.
        It computes: f(x) = clip(a * x + 0.5, 0, 1)
        """
        return torch.clamp(self.a * x + 0.5, min=0.0, max=1.0)

    def forward(self, x, train_gates=True):
        """
        Forward pass of the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (n, 1, dim).
            train_gates (bool): If True, add noise for stochastic gating (used during training).

        Returns:
            gated_x (torch.Tensor): The input features multiplied elementwise by the gate values.
            gate (torch.Tensor): The gating values (in [0,1]) for each feature.
            alpha (torch.Tensor): The raw output from the gating network (before noise and activation).
        """
        # (batch,seq_len, features_p_g, embed)
        batch , seq, feat_p_g, embed = x.size()
        x_flat = x.reshape(batch, feat_p_g*seq, embed)

        # (seq_len, batch_size, hidden_dim)
        h = self.gating_layers(x_flat)
        # Final linear transformation to produce raw gate values (alpha)
        alpha = self.alpha_layer(h)  # (seq_len, batch_size, num_features)

        # During training, add Gaussian noise to alpha to allow stochastic gating.
        if self.training or train_gates:
            noise = torch.randn_like(alpha)
            z = alpha + self.sigma * noise
        else:
            z = alpha

        # Apply the hard sigmoid function to get gate values between 0 and 1.
        gate = self.hard_sigmoid(z)  # Shape: (batch,seq, dim)
        gate = gate.view(batch, seq*feat_p_g, embed)
        gated_x = x_flat * gate
        gated_x = gated_x.reshape(batch , seq, feat_p_g, embed)

        return gated_x, gate, alpha


class GatedPerFeatureTransformer(PerFeatureTransformer):
    def __init__(
            self,
            **kwargs,  # Other parameters for PerFeatureTransformer
    ):
        """
        A PerFeatureTransformer that applies gating before its core forward pass.

        Args:
            input_dim (int): Number of features per sample.
            feature_gate_settings (dict): Settings for LearnedFeatureGating.
            sample_gate_settings (dict): Settings for LearnedSampleGating.
            **kwargs: Additional arguments for PerFeatureTransformer.
        """
        super().__init__(**kwargs)
        # Instantiate gating modules that expect batch-first data.
        self.feature_gate = LearnedFeatureGating(self.ninp,
                                                 **({"gating_hidden_dims": [50, 25], "a": 1.0, "sigma": 0.5}))
        self.sample_gate = LearnedSampleGating(self.ninp, **({"hidden_dims": [50, 25], "a": 1.0, "sigma": 0.5}))

    def _forward(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor | dict,
        # TODO(eddiebergman): Not sure if it can be None but the function seems to
        # indicate it could
        y: torch.Tensor | dict | None,
        *,
        single_eval_pos: int | None = None,
        only_return_standard_out: bool = True,
        style: torch.Tensor | None = None,
        data_dags: list[Any] | None = None,
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:
        """The core forward pass of the model.

        Args:
            x: The input data. Shape: `(seq_len, batch_size, num_features)`
            y: The target data. Shape: `(seq_len, batch_size)`
            single_eval_pos:
                The position to evaluate at. If `None`, evaluate at all positions.
            only_return_standard_out: Whether to only return the standard output.
            style: The style vector.
            data_dags: The data DAGs for each example in the batch.
            categorical_inds: The indices of categorical features.
            half_layers: Whether to use half the layers.

        Returns:
            A dictionary of output tensors.
        """
        print("xshape",x.shape)

        assert style is None
        if self.cache_trainset_representation:
            if not single_eval_pos:  # none or 0
                assert y is None
        else:
            assert y is not None
            assert single_eval_pos
        single_eval_pos_ = single_eval_pos or 0
        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:
            x = {"main": x}
        seq_len, batch_size, num_features = x["main"].shape

        if y is None:
            # TODO: check dtype.
            y = torch.zeros(
                0,
                batch_size,
                device=x["main"].device,
                dtype=x["main"].dtype,
            )

        if isinstance(y, dict):
            assert "main" in set(y.keys()), f"Main must be in input keys: {y.keys()}."
        else:
            y = {"main": y}

        for k in x:
            num_features_ = x[k].shape[2]

            # pad to multiple of features_per_group
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )

        # Splits up the input into subgroups
        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "s b (f n) -> b s f n",
                n=self.features_per_group,
            )  # s b f -> b s #groups #features_per_group

        # We have to re-work categoricals based on the subgroup they fall into.
        categorical_inds_to_use: list[list[int]] | None = None
        if categorical_inds is not None:
            new_categorical_inds = []
            n_subgroups = x["main"].shape[2]

            for subgroup in range(n_subgroups):
                subgroup_lower = subgroup * self.features_per_group
                subgroup_upper = (subgroup + 1) * self.features_per_group
                subgroup_indices = [
                    i - subgroup_lower
                    for i in categorical_inds
                    if subgroup_lower <= i < subgroup_upper
                ]
                new_categorical_inds.append(subgroup_indices)

            categorical_inds_to_use = new_categorical_inds

        for k in y:
            if y[k].ndim == 1:
                y[k] = y[k].unsqueeze(-1)
            if y[k].ndim == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos_
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert k != "main" or y[k].shape[1] == single_eval_pos_, (
                    "For main y, y must not be given for target"
                    " time steps (Otherwise the solution is leaked)."
                )
                if y[k].shape[1] == single_eval_pos_:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

        # making sure no label leakage ever happens
        y["main"][single_eval_pos_:] = torch.nan
        print("yshape",y['main'].shape)
        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del y
        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

        extra_encoders_args = {}
        if categorical_inds_to_use is not None and isinstance(
            self.encoder,
            SequentialEncoder,
        ):
            extra_encoders_args["categorical_inds"] = categorical_inds_to_use

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")



        embedded_x = einops.rearrange(
            self.encoder(
                x,
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=self.cache_trainset_representation,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x



        embedded_x, embedded_y = self.add_embeddings(
            embedded_x,
            embedded_y,
            data_dags=data_dags,
            num_features=num_features,
            seq_len=seq_len,
            cache_embeddings=(
                self.cache_trainset_representation and single_eval_pos is not None
            ),
            use_cached_embeddings=(
                self.cache_trainset_representation and single_eval_pos is None
            ),
        )
        del data_dags

        # b s f e + b s 1 e -> b s f+1 e

        embedded_input = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

        gated, gate_feat, _ = self.feature_gate(embedded_input, train_gates=self.training)
        # Apply sample gating.
        gated, gate_sample, _ = self.sample_gate(gated, train_gates=self.training)

        embedded_input = gated
        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x and y."
                "Check that you do not feed NaNs or use a NaN-handling enocder."
                "Your embedded x and y returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_y).any()=}",
            )
        del embedded_y, embedded_x

        encoder_out = self.transformer_encoder(
            (
                embedded_input
                if not self.transformer_decoder
                else embedded_input[:, :single_eval_pos_]
            ),
            single_eval_pos=single_eval_pos,
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # b s f+1 e -> b s f+1 e

        # If we are using a decoder
        if self.transformer_decoder:
            assert not half_layers
            assert encoder_out.shape[1] == single_eval_pos_

            if self.global_att_embeddings_for_compression is not None:
                # TODO: fixed number of compression tokens
                train_encoder_out = self.encoder_compression_layer(
                    self.global_att_embeddings_for_compression,
                    att_src=encoder_out[:, single_eval_pos_],
                    single_eval_pos=single_eval_pos_,
                )

            test_encoder_out = self.transformer_decoder(
                embedded_input[:, single_eval_pos_:],
                single_eval_pos=0,
                att_src=encoder_out,
            )
            encoder_out = torch.cat([encoder_out, test_encoder_out], 1)
            del test_encoder_out

        del embedded_input

        # out: s b e
        test_encoder_out = encoder_out[:, single_eval_pos_:, -1].transpose(0, 1)

        if only_return_standard_out:
            assert self.decoder_dict is not None
            output_decoded = self.decoder_dict["standard"](test_encoder_out)
        else:
            output_decoded = (
                {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
                if self.decoder_dict is not None
                else {}
            )

            # out: s b e
            train_encoder_out = encoder_out[:, :single_eval_pos_, -1].transpose(0, 1)
            output_decoded["train_embeddings"] = train_encoder_out
            output_decoded["test_embeddings"] = test_encoder_out

        return output_decoded

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True,
                        assign: bool = False) -> Any:
        # Separate gating parameters from base model parameters
        gating_state = {}
        base_state = {}
        for key, value in state_dict.items():
            if key.startswith("feature_gate.") or key.startswith("sample_gate."):
                gating_state[key] = value
            else:
                base_state[key] = value

        # Load the base model parameters (from PerFeatureTransformer)
        base_result = super().load_state_dict(base_state, strict=False)

        # Remove prefixes from gating keys so they match the modules' state dict keys.
        feature_gate_state = {
            key[len("feature_gate."):]: value
            for key, value in gating_state.items() if key.startswith("feature_gate.")
        }
        sample_gate_state = {
            key[len("sample_gate."):]: value
            for key, value in gating_state.items() if key.startswith("sample_gate.")
        }

        # Load gating modules' state dictionaries if they are present.
        if len(feature_gate_state) > 0:
            self.feature_gate.load_state_dict(feature_gate_state, strict=False)
        else:
            print(
                "Warning: No feature gate parameters found in state_dict; using random initialization for feature_gate.")

        if len(sample_gate_state) > 0:
            self.sample_gate.load_state_dict(sample_gate_state, strict=False)
        else:
            print(
                "Warning: No sample gate parameters found in state_dict; using random initialization for sample_gate.")
        # Freeze the base model parameters by setting requires_grad=False for parameters
        # not belonging to the gating modules.
        for name, param in self.named_parameters():
            if not (name.startswith("feature_gate.") or name.startswith("sample_gate.")):
                param.requires_grad = False

        return base_result
