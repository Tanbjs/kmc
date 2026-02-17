from typing import Dict, Any, Optional
from collections import OrderedDict

import torch
from torch.functional import F
from torch import nn
from torch import optim

import lightning as L
from kmc.utils.nn import NNHelper

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, config: Dict[str, Any]):
        super(Encoder, self).__init__()    
        enc_layer = OrderedDict()
        in_features = input_dim
        
        for name, cfg in config.items():
            layer = nn.Linear(in_features, cfg['units'], bias=cfg.get('bias_init', True))
            NNHelper.init_layer(layer, cfg.get('weight_init', 'xavier_normal'), 'zeros')
            enc_layer[name] = layer
            act_str = cfg.get('activation')
            activation = NNHelper.get_activation(act_str)
            if activation:
                enc_layer[f"{name}_{act_str}"] = activation
            in_features = cfg['units']            
            
        enc_layer['last_layer'] = nn.Linear(in_features, latent_dim)
        self.enc_net = nn.Sequential(enc_layer)

    def forward(self, x):
        return self.enc_net(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, config: Dict[str, Any]):
        super(Decoder, self).__init__()
        dec_layer = OrderedDict()
        in_features = latent_dim
        
        for name, cfg in config.items():
            layer = nn.Linear(in_features, cfg['units'], bias=cfg.get('bias_init', True))
            NNHelper.init_layer(layer, cfg.get('weight_init', 'xavier_normal'), 'zeros')
            dec_layer[name] = layer
            act_str = cfg.get('activation')
            activation = NNHelper.get_activation(act_str)
            if activation:
                dec_layer[f"{name}_{act_str}"] = activation
            in_features = cfg['units']            
            
        dec_layer['last_layer'] = nn.Linear(in_features, output_dim)
        self.dec_net = nn.Sequential(dec_layer)

    def forward(self, z):
        return self.dec_net(z)

class KAEc(nn.Module):
    def __init__(self, 
                 feature_dim: int, 
                 control_dim: int,
                 latent_dim: int, 
                 decoder_mode: str,
                 hidden_enc_cfg: Dict[str, Any],
                 hidden_dec_cfg: Dict[str, Any]
                 ):

        super(KAEc, self).__init__()

        # 1. Output Dimension
        self.decoder_mode = decoder_mode
        
        # 2. Encoder
        self.encoder = Encoder(feature_dim, latent_dim, hidden_enc_cfg)

        # 3. System Matrices (A, B)
        # Total Latent Dim = Output Dim (y) + Encoder Dim (phi)
        self.total_latent_dim = feature_dim + latent_dim
        self.A = nn.Linear(self.total_latent_dim, self.total_latent_dim, bias=False)
        self.B = nn.Linear(control_dim, self.total_latent_dim, bias=False)

        # Initialize A near Identity
        with torch.no_grad():
            self.A.weight.data = torch.eye(self.total_latent_dim) + 0.001 * torch.randn(self.total_latent_dim, self.total_latent_dim)
            self.B.weight.data = 0.001 * torch.randn(self.total_latent_dim, control_dim)

        # 4. Decoder
        if self.decoder_mode == 'linear':
            self.decoder = nn.Linear(self.total_latent_dim, feature_dim, bias=False)
            with torch.no_grad():
                self.decoder.weight.zero_() 
                self.decoder.weight[:, :feature_dim] = torch.eye(feature_dim)
            self.decoder.weight.requires_grad = False
            
        elif self.decoder_mode == 'nonlinear':
            self.decoder = Decoder(self.total_latent_dim, feature_dim, hidden_dec_cfg)
        else:
            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

    def lift(self, x):
        return torch.cat((x, self.encoder(x)), dim=-1)

    def forward(self, x_curr, u_curr):
        z = self.lift(x_curr)                   # 1. Lift
        z_next = self.A(z) + self.B(u_curr)     # 2. Linear Evolution: z_next = z @ A.T + u @ B.T
        x_next = self.decoder(z_next)           # 3. Decode back to physical space
        return x_next, z_next

class LitKAEc(L.LightningModule):
    def __init__(self, 
                 model_config: Dict[str, Any],
                 optimizer_config: Dict[str, Any],
                 scheduler_config: Optional[Dict[str, Any]],
                 loss_weights: Dict[str, float]):
        """
        Args:
            model_config: Dict params passed directly to KAEc (feature_dim, latent_dim, etc.)
            optimizer_config: Dict containing 'type' and 'params' (lr, weight_decay, etc.)
            loss_weights: Dict containing weights (w_rec, w_lin, w_pred, w_enc, w_dec)
        """
        super().__init__()
        self.save_hyperparameters() # Saves all dicts to self.hparams

        # 1. Initialize Model (Unpacking Config)
        # Ensure KAEc class accepts **kwargs or matches these keys
        self.model = KAEc(**self.hparams.model_config) 

    def forward(self, x_curr, u_curr):
        return self.model(x_curr, u_curr)

    def configure_optimizers(self):
        opt_cfg = self.hparams.optimizer_config
        opt_type = opt_cfg.get('type', 'adam').lower()
        opt_params = opt_cfg.get('params', {'lr': 1e-3})
        scheduler_type = self.hparams.scheduler_config.get('type') if self.hparams.scheduler_config else None
        scheduler_params = self.hparams.scheduler_config.get('params', {}) if self.hparams.scheduler_config else {}

        if opt_type == 'adam':
            optimizer = optim.Adam(self.parameters(), **opt_params)
        elif opt_type == 'lbfgs':
            optimizer = optim.LBFGS(self.parameters(), **opt_params)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        if scheduler_type.lower() == 'reducelronplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/total_loss",
                "strict": False,
            }
        }

    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "valid")

    def _shared_eval_step(self, batch, batch_idx, prefix):
        
        # Batch from AUVLazyDataset: (Batch, Seq_Len, Dim)
        weights = self.hparams.loss_weights
        x_curr, u_curr, x_next_true = batch        
        
        x0 = x_curr[:, 0, :].unsqueeze(1)                         
        x_multi_pred, z_multi_pred = self._multi_step_pred(x0, u_curr)

        # LOSS 1: Reconstruction (Physical Space)
        z_next_pred = self.model.lift(x_next_true)
        x_next_pred = self.model.decoder(z_next_pred)
        loss_recon = F.mse_loss(x_next_pred, x_next_true)

        # LOSS 2: Linear Evolution Consistency (Latent Space)
        loss_lin = F.mse_loss(z_multi_pred, z_next_pred)

        # LOSS 3: Prediction (Long-term stability)
        loss_pred = F.mse_loss(x_multi_pred, x_next_true)

        # LOSS 4: Infinity Norm (Physical Space)
        L_inf_rec = torch.mean(torch.norm(x_next_pred - x_next_true, p=float('inf'), dim=-1))
        L_inf_pred = torch.mean(torch.norm(x_multi_pred - x_next_true, p=float('inf'), dim=-1))
        L_inf = L_inf_rec + L_inf_pred

        # --- 3. Regularization ---
        loss_enc_reg = sum(p.pow(2).sum() for n, p in self.model.encoder.named_parameters() if 'weight' in n)
        loss_dec_reg = 0.0
        if self.model.decoder_mode == 'nonlinear':
            loss_dec_reg = sum(p.pow(2).sum() for n, p in self.model.decoder.named_parameters() if 'weight' in n)

        # --- Total Weighted Loss ---
        total_loss = (weights.get('w_rec', 1.0) * loss_recon) + \
                     (weights.get('w_lin', 1.0) * loss_lin) + \
                     (weights.get('w_pred', 1.0) * loss_pred) + \
                     (weights.get('w_inf', 0.0) * L_inf) + \
                     (weights.get('w_enc_reg', 0.0) * loss_enc_reg) + \
                     (weights.get('w_dec_reg', 0.0) * loss_dec_reg)

        # Logging
        self.log(f'{prefix}/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/recon_loss', loss_recon, on_step=False, on_epoch=True)
        self.log(f'{prefix}/pred_loss', loss_pred, on_step=False, on_epoch=True)
        self.log(f'{prefix}/lin_loss', loss_lin, on_step=False, on_epoch=True)
        self.log(f'{prefix}/inf_norm_loss', L_inf, on_step=False, on_epoch=True)
        return total_loss
    
    def _multi_step_pred(self, x_init: torch.Tensor, U: torch.Tensor):
        """
        x_init: (Batch, 1, Dim) - State at t=0
        U:      (Batch, Steps, Dim) - Control sequence
        Returns: 
            x_pred_seq: (Batch, Steps, Dim) - Predicted sequence using Delta strategy
            z_pred_seq: (Batch, Steps, Dim) - Latent sequence
        """
        steps = U.shape[1]
        z = self.model.lift(x_init)      # (B, 1, Latent)
        x_pred_stack = []
        z_pred_stack = []
        
        for i in range(steps):
            u_step = U[:, i, :].unsqueeze(1)  # (B, 1, Input_Dim)
            # --- Linear Dynamics (Latent Space) ---
            # z_{k+1} = A z_k + B u_k
            z_next = self.model.A(z) + self.model.B(u_step) 
            
            # Store results
            x_next = self.model.decoder(z_next)
            z_pred_stack.append(z_next)
            x_pred_stack.append(x_next)
            
            # Update for next step
            z = z_next       

        return torch.cat(x_pred_stack, dim=1), torch.cat(z_pred_stack, dim=1)