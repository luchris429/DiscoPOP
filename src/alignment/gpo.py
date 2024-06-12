from typing import Tuple

import torch
import torch.nn.functional as F

from trl import DPOTrainer


class GPOTrainer(DPOTrainer):
    def __init__(self, *args, func=lambda: None, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def gpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.loss_type == "epo":
            return self.func(
                self,
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "aqfl":
            return self.adaptive_quantile_feedback_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "dbaql":
            return self.dynamic_blended_adaptive_quantile_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "aql":
            return self.adaptive_quantile_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "padll":
            return self.performance_adaptive_decay_logistic_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "cell":
            return self.combined_exp_logistic_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "lrml":
            return self.log_ratio_modulated_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "pfl":
            return self.policy_focused_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type in ["sigmoid", "dpo"]:
            return self.sigmoid_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "hinge":
            return self.hinge_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "ipo":
            return self.ipo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "kto_pair":
            return self.kto_pair_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        elif self.loss_type == "bco_pair":
            return self.bco_pair_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair', 'epo']"
            )

    def dynamic_blended_adaptive_quantile_loss_old(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        import torch.nn.functional as F

        # Constants for the loss function
        starting_quantile = 0.5
        quantile_adapt_rate = 0.01
        temperature = 0.9
        dynamic_blend_rate = 1.0

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        logits_variability = logits.var()

        # Calculate an adaptive quantile based on a moving target
        starting_quantile + quantile_adapt_rate * (torch.sigmoid(logits.mean()) - starting_quantile)

        # Calculate dynamic blending coefficient based on logits variability
        dynamic_blend_coeff = torch.sigmoid(logits_variability) * dynamic_blend_rate

        # Prepare components of the blended loss
        logistic_loss = -F.logsigmoid(self.beta * logits / temperature)
        exp_loss = torch.exp(-self.beta * logits * temperature)

        # Blend the losses dynamically
        losses = dynamic_blend_coeff * logistic_loss + (1 - dynamic_blend_coeff) * exp_loss
        return losses

    def dynamic_blended_adaptive_quantile_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        import torch.nn.functional as F

        # Constants for the loss function
        temperature = 0.9
        dynamic_blend_rate = 1.0

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        logits = logits * self.beta

        logits_variability = (logits / 0.05).var()

        # Calculate dynamic blending coefficient based on logits variability
        dynamic_blend_coeff = torch.sigmoid(logits_variability) * dynamic_blend_rate

        # Prepare components of the blended loss
        logistic_loss = -F.logsigmoid(logits / temperature)
        exp_loss = torch.exp(-logits * temperature)

        # Blend the losses dynamically
        losses = dynamic_blend_coeff * logistic_loss + (1 - dynamic_blend_coeff) * exp_loss
        return losses

    def adaptive_quantile_loss_old(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        percentile = 0.5  # Start with the median quantile
        moving_quantile_weight = 0.01  # Weight for updating the moving quantile
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        moving_quantile = percentile + moving_quantile_weight * (torch.sigmoid(logits.mean()) - percentile)

        quantile_weights = torch.sigmoid(-self.beta * (logits - moving_quantile))

        logistic_losses = -F.logsigmoid(self.beta * logits)
        hinge_losses = torch.relu(1 - self.beta * logits)

        # Blend the logistic and hinge losses based on the dynamic quantile weight
        losses = quantile_weights * logistic_losses + (1 - quantile_weights) * hinge_losses
        return losses

    def adaptive_quantile_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        percentile = 0.5  # Start with the median quantile
        moving_quantile_weight = 0.01  # Weight for updating the moving quantile
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        logits = logits * self.beta

        moving_quantile = percentile + moving_quantile_weight * (torch.sigmoid((logits / 0.05).mean()) - percentile)

        quantile_weights = torch.sigmoid(-(logits / 0.05 - moving_quantile))

        logistic_losses = -F.logsigmoid(logits)
        hinge_losses = torch.relu(1 - logits)

        # Blend the logistic and hinge losses based on the dynamic quantile weight
        losses = quantile_weights * logistic_losses + (1 - quantile_weights) * hinge_losses
        return losses

    def adaptive_quantile_feedback_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        import torch.nn.functional as F

        quantile_update_rate = 0.05
        distance_scale = 0.1

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        logits = logits * self.beta

        logits_std = (logits / 0.05).std()

        adaptive_quantile = logits_std * torch.sigmoid(-logits / 0.05).mean()
        adaptive_quantile += quantile_update_rate * (torch.sigmoid((logits / 0.05).mean()) - adaptive_quantile)

        distance_from_quantile = (logits / 0.05 - adaptive_quantile).abs()
        blend_rate = torch.sigmoid(distance_scale * distance_from_quantile)

        logistic_losses = -F.logsigmoid(logits)
        hinge_losses = torch.relu(1 - logits)

        losses = blend_rate * logistic_losses + (1 - blend_rate) * hinge_losses
        return losses

    def adaptive_quantile_feedback_loss_old(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        import torch.nn.functional as F

        quantile_update_rate = 0.05
        distance_scale = 0.1

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        logits_std = logits.std()

        adaptive_quantile = logits_std * torch.sigmoid(-logits).mean()
        adaptive_quantile += quantile_update_rate * (torch.sigmoid(logits.mean()) - adaptive_quantile)

        distance_from_quantile = (logits - adaptive_quantile).abs()
        blend_rate = torch.sigmoid(distance_scale * distance_from_quantile)

        logistic_losses = -F.logsigmoid(self.beta * logits)
        hinge_losses = torch.relu(1 - self.beta * logits)

        losses = blend_rate * logistic_losses + (1 - blend_rate) * hinge_losses
        return losses

    def performance_adaptive_decay_logistic_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        base_decay = 0.9
        mismatch_penalty = 0.5  # Penalty decay for mismatched choices

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        mismatches = (logits < 0).float()  # Identify mismatches

        adaptive_decay = base_decay * (1 - mismatches * mismatch_penalty)
        weighted_losses = adaptive_decay * -F.logsigmoid(self.beta * logits)
        return weighted_losses

    def combined_exp_logistic_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        exp_losses = torch.exp(-self.beta * logits)
        log_losses = -F.logsigmoid(self.beta * logits)
        # Combine the losses with a tunable mixing coefficient
        alpha = 0.5
        losses = alpha * exp_losses + (1 - alpha) * log_losses
        return losses

    def log_ratio_modulated_loss_old(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        # Modulate the mixing coefficient based on the log ratio magnitudes
        log_ratio_modulation = torch.sigmoid(logits)
        logistic_component = -F.logsigmoid(self.beta * logits)
        exp_component = torch.exp(-self.beta * logits)
        # Blend between logistic and exponential component based on log ratio modulation
        losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
        return losses

    def log_ratio_modulated_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        tau = 0.05
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        logits = logits * self.beta
        # Modulate the mixing coefficient based on the log ratio magnitudes
        log_ratio_modulation = torch.sigmoid(logits / tau)
        logistic_component = -F.logsigmoid(logits)
        exp_component = torch.exp(-logits)
        # Blend between logistic and exponential component based on log ratio modulation
        losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
        return losses

    def policy_focused_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        focus_scale = 2.0  # Scale to emphasize or de-emphasize based on the correctness of predictions

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        logits = logits * self.beta

        is_correct = policy_chosen_logps > policy_rejected_logps

        logistic_losses = -F.logsigmoid(logits)
        hinge_losses = torch.relu(1 - logits)

        focused_loss = torch.where(
            is_correct,
            logistic_losses / focus_scale,  # De-emphasize correct predictions
            hinge_losses * focus_scale,  # Emphasize incorrect predictions
        )
        return focused_loss

    def policy_focused_loss_old(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        focus_scale = 2.0  # Scale to emphasize or de-emphasize based on the correctness of predictions

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        is_correct = policy_chosen_logps > policy_rejected_logps

        logistic_losses = -F.logsigmoid(logits)
        hinge_losses = torch.relu(1 - logits)

        focused_loss = torch.where(
            is_correct,
            logistic_losses / focus_scale,  # De-emphasize correct predictions
            hinge_losses * focus_scale,  # Emphasize incorrect predictions
        )
        return focused_loss

    def sigmoid_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )
        return losses

    def hinge_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = torch.relu(1 - self.beta * logits)
        return losses

    def ipo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = (logits - 1 / (2 * self.beta)) ** 2
        return losses

    def kto_pair_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )
        return losses

    def bco_pair_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        self.running.update(rewards)
        delta = self.running.mean

        losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
            -(self.beta * rejected_logratios - delta)
        )
        return losses

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        policy_chosen_logps = policy_chosen_logps.to(self.accelerator.device)
        policy_rejected_logps = policy_rejected_logps.to(self.accelerator.device)
        reference_chosen_logps = reference_chosen_logps.to(self.accelerator.device)
        reference_rejected_logps = reference_rejected_logps.to(self.accelerator.device)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        losses = self.gpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        return losses, chosen_rewards, rejected_rewards

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.

        # if self.loss_type == "sigmoid":
        #     losses = (
        #         -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
        #         - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        #     )
        # elif self.loss_type == "hinge":
        #     losses = torch.relu(1 - self.beta * logits)
        # elif self.loss_type == "ipo":
        #     # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
        #     losses = (logits - 1 / (2 * self.beta)) ** 2
        # elif self.loss_type == "kto_pair":
        #     # eqn (7) of the HALOs paper
        #     chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        #     rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        #     chosen_logratios = policy_chosen_logps - reference_chosen_logps
        #     rejected_logratios = policy_rejected_logps - reference_rejected_logps
        #     # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
        #     losses = torch.cat(
        #         (
        #             1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
        #             1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
        #         ),
        #         0,
        #     )
        # elif self.loss_type == "bco_pair":
        #     chosen_logratios = policy_chosen_logps - reference_chosen_logps
        #     rejected_logratios = policy_rejected_logps - reference_rejected_logps

        #     chosen_rewards = self.beta * chosen_logratios
        #     rejected_rewards = self.beta * rejected_logratios
        #     rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        #     self.running.update(rewards)
        #     delta = self.running.mean

        #     losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
        #         -(self.beta * rejected_logratios - delta)
        #     )
        # else:
        #     raise ValueError(
        #         f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair', 'gpo']"
        #     )
