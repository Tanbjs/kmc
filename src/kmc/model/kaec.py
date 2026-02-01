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
                 hidden_dec_cfg: Dict[str, Any], 
                 target_indices: Optional[Any] = None):

        super(KAEc, self).__init__()

        # 1. Output Dimension
        if target_indices is not None:
            self.target_indices = target_indices
            self.output_dim = len(target_indices) 
        else:
            self.target_indices = None
            self.output_dim = feature_dim
        
        self.decoder_mode = decoder_mode
        
        # 2. Encoder
        self.encoder = Encoder(feature_dim, latent_dim, hidden_enc_cfg)

        # 3. System Matrices (A, B)
        # Total Latent Dim = Output Dim (y) + Encoder Dim (phi)
        self.total_latent_dim = self.output_dim + latent_dim
        self.A = nn.Linear(self.total_latent_dim, self.total_latent_dim, bias=False)
        self.B = nn.Linear(control_dim, self.total_latent_dim, bias=False)

        # Initialize A near Identity
        with torch.no_grad():
            self.A.weight.data = torch.eye(self.total_latent_dim) + 0.001 * torch.randn(self.total_latent_dim, self.total_latent_dim)
            self.B.weight.data = 0.001 * torch.randn(self.total_latent_dim, control_dim)

        # 4. Decoder
        if self.decoder_mode == 'linear':
            self.decoder = nn.Linear(self.total_latent_dim, self.output_dim, bias=False)
            with torch.no_grad():
                self.decoder.weight.zero_() 
                self.decoder.weight[:, :self.output_dim] = torch.eye(self.output_dim)
            self.decoder.weight.requires_grad = False
            
        elif self.decoder_mode == 'nonlinear':
            self.decoder = Decoder(self.total_latent_dim, self.output_dim, hidden_dec_cfg)
        else:
            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

    def lift(self, x):
        if self.target_indices is not None:
            y = x[:, self.target_indices]
        else:
            y = x
        return torch.cat((y, self.encoder(x)), dim=-1)

    def forward(self, x_curr, u_curr):
        z = self.lift(x_curr)                   # 1. Lift
        z_next = self.A(z) + self.B(u_curr)     # 2. Linear Evolution: z_next = z @ A.T + u @ B.T
        x_next = self.decoder(z_next)           # 3. Decode back to physical space
        return x_next, z_next

class LitKAEc(L.LightningModule):
    def __init__(self, 
                 input_dim: int,
                 control_dim: int,       
                 latent_dim: int,
                 decoder_mode: str,
                 hidden_enc_cfg: Dict,
                 hidden_dec_cfg: Dict,
                 target_indices: Optional[Any] = None,
                 learning_rate: float = 1e-3,
                 w_rec: float = 1.0,
                 w_pred: float = 1.0,    
                 w_lin: float = 1.0,
                 w_enc_reg: float = 1.0,
                 w_dec_reg: float = 1.0):    

        super().__init__()
        self.save_hyperparameters()

        self.model = KAEc(
            feature_dim=input_dim,
            control_dim=control_dim,
            latent_dim=latent_dim,
            decoder_mode=decoder_mode,
            hidden_enc_cfg=hidden_enc_cfg,
            hidden_dec_cfg=hidden_dec_cfg,
            target_indices=target_indices
        )

    def forward(self, x_curr, u_curr):
        return self.model(x_curr, u_curr)

    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, "valid")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/total_loss"
            }
        }

    def _shared_eval_step(self, batch, batch_idx, prefix):
        # Batch assumed format: (x_k, u_k, x_k+1)
        x_curr, u_curr, x_next = batch        
        
        # 1. Forward Pass
        x_next_pred, z_next_pred = self.model(x_curr, u_curr)

        # Filter target indices for Ground Truth comparison
        if self.model.target_indices is not None:
            x_next_target = x_next[:, self.model.target_indices]
        else:
            x_next_target = x_next

        # --- LOSS 1: Reconstruction Loss ---
        loss_recon = F.mse_loss(x_next_pred, x_next_target)

        # --- LOSS 2: Linear Evolution Loss ---
        with torch.no_grad():       
            z_next_real = self.model.lift(x_next)
        loss_lin = F.mse_loss(z_next_pred, z_next_real)

        # --- LOSS 3: Prediction Loss (Multi-step) ---
        # Note: Using small p=1 step here based on batch format, 
        if u_curr.ndim == 2: # (Batch, Dim) -> (Batch, 1, Dim)
            u_in = u_curr.unsqueeze(1)
            x_in = x_curr.unsqueeze(1)
        else:
            u_in = u_curr
            x_in = x_curr

        x_multi_pred = self._multi_step_pred(x_in, u_in)

        loss_pred = F.mse_loss(x_multi_pred[:, -1, :], x_next_target)

        # --- LOSS 4 & 5: L2 Regularization (Sum of Squares) ---
        loss_enc_reg = sum(p.pow(2).sum() for n, p in self.model.encoder.named_parameters() if 'weight' in n)
        if self.model.decoder_mode == 'nonlinear':
            loss_dec_reg = sum(p.pow(2).sum() for n, p in self.model.decoder.named_parameters() if 'weight' in n)
        elif self.model.decoder_mode == 'linear':
            loss_dec_reg = 0.0

        # Total Weighted Loss        
        total_loss = (self.hparams.w_rec * loss_recon) + \
                     (self.hparams.w_lin * loss_lin) + \
                     (self.hparams.w_pred * loss_pred) + \
                     (self.hparams.w_enc_reg * loss_enc_reg) + \
                     (self.hparams.w_dec_reg * loss_dec_reg)

        # Logging
        self.log(f'{prefix}/total_loss', total_loss, prog_bar=True)
        self.log(f'{prefix}/recon_loss', loss_recon)
        self.log(f'{prefix}/lin_loss', loss_lin)
        self.log(f'{prefix}/pred_loss', loss_pred)
        self.log(f'{prefix}/enc_reg_loss', loss_enc_reg)
        self.log(f'{prefix}/dec_reg_loss', loss_dec_reg)
        return total_loss
    
    def _multi_step_pred(self, X: torch.Tensor, U: torch.Tensor):
        """
        Helper for multi-step prediction returning PHYSICAL state (x).
        X: (Batch, 1, Dim)
        U: (Batch, Steps, Dim)
        """
        steps = U.shape[1]
        x_curr = X[:, 0, :] # Initial State
        
        # 1. Lift to z
        z = self.model.lift(x_curr)
        
        x_pred_stack = []
        
        for i in range(steps):
            # 2. Evolve z
            # z_next = z @ A.T + u @ B.T
            u_step = U[:, i, :]
            z = self.model.A(z) + self.model.B(u_step)
            
            # 3. Decode back to x (for loss comparison)
            x_step = self.model.decoder(z)
            x_pred_stack.append(x_step)
            
        return torch.stack(x_pred_stack, dim=1)