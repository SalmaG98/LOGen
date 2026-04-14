import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import LightningDataModule
from logen.models.logen import LOGen_models
from logen.models.dit3d import DiT3D_models
from logen.modules.metrics import ChamferDistance, EMD, RMSE, JSD
from logen.modules.three_d_helpers import build_two_point_clouds


class FlowMatcher(LightningModule):
    """
    Flow Matching module for generative modeling.
    
    Flow matching learns a velocity field that moves from noise to data along straight paths.
    This is simpler and often more efficient than diffusion models.
    """
    
    def __init__(self, hparams: dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        
        self.t_steps = self.hparams['flow'].get('t_steps', 1000)  # Not used directly but kept for compatibility
        self.s_steps = self.hparams['flow'].get('s_steps', 50)  # Number of steps for ODE solving
        self.ode_solver = self.hparams['flow'].get('ode_solver', 'euler')  # 'euler' or 'rk4'
        
        conditioning_type = self.hparams['model']['conditioning']
        point_embeddings = self.hparams['model']['embeddings']
        model_size = self.hparams['model']['size']
        num_classes = self.hparams['model']['num_classes']
        num_total_conditions = self.hparams['model']['num_conditions']
        num_cyclic_conditions = self.hparams['model']['cyclic_conditions']
        num_linear_conditions = self.hparams['model']['linear_conditions']
        map_shape_to_one = self.hparams['model']['map_shape_to_one']
        self.in_channels = self.hparams['model']['in_channels']
        
        self.model = self.model_factory(
            conditioning_type, point_embeddings, model_size, 
            num_cyclic_conditions, num_linear_conditions, 
            map_shape_to_one, num_classes, self.in_channels, num_total_conditions
        )
        
        self.chamfer_distance = ChamferDistance()
        self.emd = EMD()
        self.rmse = RMSE()
        self.jsd = JSD()
        
        self.w_uncond = self.hparams['train']['uncond_w']
        self.visualize = self.hparams['flow'].get('visualize', False)

    def model_factory(self, conditioning_type, point_embeddings, model_size, 
                     num_cyclic_conditions, num_linear_conditions, 
                     map_shape_to_one, num_classes, in_channels, num_total_conditions):
        """Factory method to create the appropriate model architecture."""
        factory = None
        if conditioning_type == 'logen':
            factory = LOGen_models
        elif conditioning_type == 'dit3d':
            factory = DiT3D_models
        model = factory[model_size]
        return model(
            num_classes=num_classes, 
            in_channels=in_channels, 
            num_cyclic_conditions=num_cyclic_conditions, 
            num_linear_conditions=num_linear_conditions, 
            map_shape_to_one=map_shape_to_one, 
            num_total_conditions=num_total_conditions
        )

    def interpolate(self, x0, x1, t):
        """
        Linear interpolation between noise (x0) and data (x1).
        x(t) = (1 - t) * x0 + t * x1
        
        Args:
            x0: Noise tensor (shape: B, C, N)
            x1: Data tensor (shape: B, C, N)
            t: Time value(s) in [0, 1] (scalar or tensor of shape B)
        
        Returns:
            Interpolated tensor at time t
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=x0.device, dtype=x0.dtype)
        
        # Reshape t for broadcasting
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Expand t to match batch size if needed
        if t.shape[0] == 1 and x0.shape[0] > 1:
            t = t.expand(x0.shape[0])
        
        t = t.view(-1, 1, 1)  # Shape: (B, 1, 1)
        return (1 - t) * x0 + t * x1

    def get_velocity(self, x0, x1):
        """
        Compute the velocity field (target for training).
        v(t) = dx/dt = x1 - x0
        
        Args:
            x0: Noise tensor (shape: B, C, N)
            x1: Data tensor (shape: B, C, N)
        
        Returns:
            Velocity tensor (same shape as x0 and x1)
        """
        return x1 - x0

    def classfree_forward(self, x_t, t, x_class, x_cond):
        """
        Classifier-free guidance: interpolate between conditional and unconditional predictions.
        """
        # Conditional prediction
        v_c = self.forward(x_t, t, x_class, x_cond, force_dropout=False)
        # Unconditional prediction
        v_uc = self.forward(x_t, t, x_class, x_cond, force_dropout=True)
        
        # Interpolate
        return v_uc + self.w_uncond * (v_c - v_uc)

    def visualize_step_t(self, x_t, gt_pts, pcd):
        """Visualize intermediate steps during generation."""
        points = x_t.detach().cpu().numpy()
        points = np.concatenate((points, gt_pts.detach().cpu().numpy()), axis=0)

        pcd.points = o3d.utility.Vector3dVector(points)
        
        colors = np.ones((len(points), 3))
        colors[:len(gt_pts)] = [1., .3, .3]
        colors[len(gt_pts):] = [.3, 1., .3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def euler_step(self, x_t, t, dt, x_class, x_cond, mask):
        """
        Single Euler step for ODE integration.
        x_{t+dt} = x_t + v(x_t, t) * dt
        """
        v_t = self.classfree_forward(x_t, t, x_class, x_cond)
        x_next = x_t + v_t * dt
        x_next = x_next * mask  # Apply mask
        return x_next

    def rk4_step(self, x_t, t, dt, x_class, x_cond, mask):
        """
        Single RK4 (Runge-Kutta 4th order) step for ODE integration.
        More accurate than Euler but more expensive.
        """
        # k1 = v(x_t, t)
        k1 = self.classfree_forward(x_t, t, x_class, x_cond)
        
        # k2 = v(x_t + dt/2 * k1, t + dt/2)
        x_mid1 = x_t + (dt / 2) * k1
        x_mid1 = x_mid1 * mask
        t_mid = t + dt / 2
        k2 = self.classfree_forward(x_mid1, t_mid, x_class, x_cond)
        
        # k3 = v(x_t + dt/2 * k2, t + dt/2)
        x_mid2 = x_t + (dt / 2) * k2
        x_mid2 = x_mid2 * mask
        k3 = self.classfree_forward(x_mid2, t_mid, x_class, x_cond)
        
        # k4 = v(x_t + dt * k3, t + dt)
        x_end = x_t + dt * k3
        x_end = x_end * mask
        t_end = t + dt
        k4 = self.classfree_forward(x_end, t_end, x_class, x_cond)
        
        # x_{t+dt} = x_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        x_next = x_t + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x_next = x_next * mask
        return x_next

    def sample_loop(self, x_t, x_class, x_cond, mask):
        """
        Generate samples by integrating the ODE from t=0 to t=1.
        """
        # Time steps: integrate from 0 to 1
        timesteps = np.linspace(0, 1, self.s_steps + 1)
        dt = 1.0 / self.s_steps
        
        solver_step = self.rk4_step if self.ode_solver == 'rk4' else self.euler_step
        
        for i in tqdm(range(len(timesteps) - 1)):
            t = torch.ones(x_t.shape[0], device=x_t.device) * timesteps[i]
            x_t = solver_step(x_t, t, dt, x_class, x_cond, mask)
        
        return x_t

    def p_losses(self, v_pred, v_true):
        """
        Compute L2 loss between predicted and target velocity.
        """
        return F.mse_loss(v_pred, v_true)

    def forward(self, x, t, y, c, force_dropout=False):
        """Forward pass through the model."""
        out = self.model(x, t, y, c, force_dropout)
        return out

    def training_step(self, batch: dict, batch_idx):
        """
        Training step for flow matching.
        
        1. Sample random noise x0
        2. Get data x1
        3. Sample random time t in [0, 1]
        4. Interpolate: x_t = (1-t)*x0 + t*x1
        5. Predict velocity: v_pred = model(x_t, t, ...)
        6. Compute target velocity: v_true = x1 - x0
        7. Compute loss as MSE(v_pred, v_true)
        """
        # Get data
        x_object = batch['pcd_object'].cuda()
        padding_mask = batch['padding_mask'][:, None, :]
        
        # Sample random noise
        x_noise = torch.randn(x_object.shape, device=self.device) * padding_mask
        
        # Sample random time uniformly in [0, 1]
        batch_size = x_object.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Interpolate between noise and data
        x_t = self.interpolate(x_noise, x_object, t) * padding_mask
        
        # Compute target velocity
        v_target = self.get_velocity(x_noise, x_object) * padding_mask
        
        # Get conditioning information
        x_center = batch['center']
        x_size = batch['size']
        x_orientation = batch['orientation']
        x_class = batch['class']
        
        # Prepare conditions (same as in diffuser)
        if self.hparams['model']['cyclic_conditions'] > 0:
            if self.hparams['model']['relative_angles'] == True:
                x_cond = torch.cat((x_center, x_size), -1)
            else:
                if self.hparams['model']['linear_conditions'] > 0:
                    x_cond = torch.cat((torch.hstack((x_center[:, 0][:, None], x_orientation)), 
                                       torch.hstack((x_center[:, 1:], x_size))), -1)
                else:
                    x_cond = torch.cat((torch.hstack((x_center[:, 0][:, None], x_orientation)), 
                                       x_center[:, 1:]), -1)
        else:
            x_cond = torch.hstack((x_center, x_size, x_orientation))
        
        # Predict velocity
        v_pred = self.forward(x_t, t, x_class, x_cond) * padding_mask
        
        # Compute loss
        loss_mse = self.p_losses(v_pred, v_target)
        loss_mean = (v_pred.mean()) ** 2
        loss_std = (v_pred.std() - 1.) ** 2
        loss = loss_mse + self.hparams['flow'].get('reg_weight', 0.0) * (loss_mean + loss_std)
        
        # Logging
        std_error = (v_pred - v_target) ** 2
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss_mean', loss_mean)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        self.log('train/var', std_error.var())
        self.log('train/std', std_error.std())
        
        return loss

    def validation_step(self, batch: dict, batch_idx):
        """Validation step: generate samples and compute metrics."""
        self.model.eval()
        with torch.no_grad():
            x_object = batch['pcd_object']
            
            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']
            
            # Prepare conditions
            if self.hparams['model']['cyclic_conditions'] > 0:
                if self.hparams['model']['relative_angles'] == True:
                    x_cond = torch.cat((x_center, x_size), -1)
                else:
                    if self.hparams['model']['linear_conditions'] > 0:
                        x_cond = torch.cat((torch.hstack((x_center[:, 0][:, None], x_orientation)), 
                                           torch.hstack((x_center[:, 1:], x_size))), -1)
                    else:
                        x_cond = torch.cat((torch.hstack((x_center[:, 0][:, None], x_orientation)), 
                                           x_center[:, 1:]), -1)
            else:
                x_cond = torch.hstack((x_center, x_size, x_orientation))
            
            padding_mask = batch['padding_mask']
            
            # Generate samples from noise (t=0 is noise, t=1 is data)
            x_noise = torch.randn(x_object.shape, device=self.device)
            x_gen_eval = self.sample_loop(x_noise, batch['class'], x_cond, padding_mask[:, None, :]).permute(0, 2, 1).squeeze(0)
            x_object = x_object.permute(0, 2, 1).squeeze(0)
            
            # Compute metrics
            for pcd_index in range(batch['num_points'].shape[0]):
                mask = padding_mask[pcd_index].int() == True
                object_pcd = x_object[pcd_index][mask]
                genrtd_pcd = x_gen_eval[pcd_index][mask]
                
                object_points = object_pcd[:, :3]
                genrtd_points = genrtd_pcd[:, :3]
                
                pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=genrtd_points, object_pcd=object_points)
                
                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.emd.update(object_points, genrtd_points)
                self.jsd.update(object_points, genrtd_points)
                
                if self.in_channels == 4:  # Measure error of intensity
                    object_intensity = object_pcd[:, 3]
                    genrtd_intensity = genrtd_pcd[:, 3]
                    self.rmse.update(object_intensity, genrtd_intensity)
        
        cd_mean, cd_std = self.chamfer_distance.compute()
        emd_mean, emd_std = self.emd.compute()
        rmse_mean, rmse_std = self.rmse.compute()
        jsd_mean, jsd_std = self.jsd.compute()
        
        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/emd_mean', emd_mean, on_step=True)
        self.log('val/emd_std', emd_std, on_step=True)
        self.log('val/intensity_mean', rmse_mean)
        self.log('val/intensity_std', rmse_std)
        self.log('val/jsd_mean', jsd_mean)
        self.log('val/jsd_std', jsd_std)
        
        return {
            'val/cd_mean': cd_mean, 'val/cd_std': cd_std,
            'val/emd_mean': emd_mean, 'val/emd_std': emd_std,
            'val/intensity_mean': rmse_mean, 'val/intensity_std': rmse_std,
            'val/jsd_mean': jsd_mean, 'val/jsd_std': jsd_std,
        }

    def test_step(self, batch: dict, batch_idx):
        """Test step: same as validation but logs to test/."""
        self.model.eval()
        with torch.no_grad():
            x_object = batch['pcd_object']
            
            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']
            
            # Prepare conditions
            if self.hparams['model']['cyclic_conditions'] > 0:
                if self.hparams['model']['relative_angles'] == True:
                    x_cond = torch.cat((x_center, x_size), -1)
                else:
                    if self.hparams['model']['linear_conditions'] > 0:
                        x_cond = torch.cat((torch.hstack((x_center[:, 0][:, None], x_orientation)), 
                                           torch.hstack((x_center[:, 1:], x_size))), -1)
                    else:
                        x_cond = torch.cat((torch.hstack((x_center[:, 0][:, None], x_orientation)), 
                                           x_center[:, 1:]), -1)
            else:
                x_cond = torch.hstack((x_center, x_size, x_orientation))
            
            padding_mask = batch['padding_mask']
            
            # Generate samples
            x_noise = torch.randn(x_object.shape, device=self.device)
            x_gen_eval = self.sample_loop(x_noise, batch['class'], x_cond, padding_mask[:, None, :]).permute(0, 2, 1).squeeze(0)
            x_object = x_object.permute(0, 2, 1).squeeze(0)
            
            # Compute metrics
            for pcd_index in range(batch['num_points'].shape[0]):
                mask = padding_mask[pcd_index].int() == True
                object_pcd = x_object[pcd_index].squeeze(0)[mask]
                genrtd_pcd = x_gen_eval[pcd_index].squeeze(0)[mask]
                
                object_points = object_pcd[:, :3]
                genrtd_points = genrtd_pcd[:, :3]
                
                pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=genrtd_points, object_pcd=object_points)
                
                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.emd.update(object_points, genrtd_points)
                self.jsd.update(object_points, genrtd_points)
                
                if self.in_channels == 4:  # Measure error of intensity
                    object_intensity = object_pcd[:, 3]
                    genrtd_intensity = genrtd_pcd[:, 3]
                    self.rmse.update(object_intensity, genrtd_intensity)
        
        cd_mean, cd_std = self.chamfer_distance.compute()
        emd_mean, emd_std = self.emd.compute()
        rmse_mean, rmse_std = self.rmse.compute()
        jsd_mean, jsd_std = self.jsd.compute()
        
        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/emd_mean', emd_mean, on_step=True)
        self.log('test/emd_std', emd_std, on_step=True)
        self.log('test/intensity_mean', rmse_mean, on_step=True)
        self.log('test/intensity_std', rmse_std, on_step=True)
        self.log('val/jsd_mean', jsd_mean)
        self.log('val/jsd_std', jsd_std)
        torch.cuda.empty_cache()
        
        return {
            'test/cd_mean': cd_mean, 'test/cd_std': cd_std,
            'test/emd_mean': emd_mean, 'test/emd_std': emd_std,
            'test/intensity_mean': rmse_mean, 'test/intensity_std': rmse_std,
            'val/jsd_mean': jsd_mean, 'val/jsd_std': jsd_std
        }

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        return [optimizer]
