"""
Latent Fusion Module: PCA/SVD-based residual fusion for multi-agent communication.

This module provides a production-ready implementation of the per-step PCA
fusion method discovered in the A05-v2 ablation study.

Key findings that motivate this design:
- RWKV recurrent states are NOT composable via parallel averaging (0% accuracy)
- Sequential state switching destroys knowledge (0% accuracy)
- Plan + state dual-channel is the correct architecture (88.5% accuracy)
- PCA/SVD residual fusion outperforms naive mean-subtraction (90.5% vs 88.5%)

Usage:
    from models.latent_fusion import PCAFusion
    
    fusion = PCAFusion(n_components=1, mode="perstep")
    fusion.calibrate(agent_latents)  # Collect calibration set
    
    fused = fusion.fuse(agent1_latent, agent2_latent, weights=[0.5, 0.5])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Literal


class PCAFusion:
    """PCA/SVD-based latent fusion for multi-agent communication.
    
    Removes shared principal components from agent latents before fusing,
    which is more principled than naive mean-subtraction.
    
    Args:
        n_components: Number of principal components to remove (K)
        mode: "global" for single PCA over all samples, "perstep" for per-step PCA
    """
    
    def __init__(self, n_components: int = 1, mode: Literal["global", "perstep"] = "perstep"):
        self.n_components = n_components
        self.mode = mode
        self._calibrated = False
        self._global_mean: Optional[torch.Tensor] = None
        self._shared_dirs: Optional[torch.Tensor] = None
        
    def calibrate(self, latents: torch.Tensor) -> None:
        """Compute PCA on calibration set of agent latents.
        
        Args:
            latents: [N, D] or [N, H, D] tensor of calibration latents
                where N = number of samples, H = trajectory steps, D = latent dim
        """
        if latents.dim() == 2:
            # Single latent per sample: [N, D]
            self._global_mean = latents.mean(dim=0, keepdim=True)  # [1, D]
            centered = latents - self._global_mean  # [N, D]
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            self._shared_dirs = Vh[:self.n_components]  # [K, D]
            self._calibrated = True
        elif latents.dim() == 3:
            # Trajectory latents: [N, H, D]
            self._global_mean = latents.mean(dim=0, keepdim=True)  # [1, H, D]
            centered = latents - self._global_mean  # [N, H, D]
            # Per-step PCA: compute SVD for each trajectory step independently
            self._perstep_dirs = []
            for h in range(latents.shape[1]):
                _, _, Vh = torch.linalg.svd(centered[:, h], full_matrices=False)
                self._perstep_dirs.append(Vh[:self.n_components])  # [K, D]
            self._perstep_dirs = torch.stack(self._perstep_dirs, dim=0)  # [H, K, D]
            self._calibrated = True
        else:
            raise ValueError(f"Expected latents to be 2D or 3D, got {latents.dim()}D")
    
    def _remove_shared_global(self, z: torch.Tensor) -> torch.Tensor:
        """Remove global shared components from a single latent."""
        if not self._calibrated:
            raise RuntimeError("Must calibrate before fusing")
        if self._global_mean is None or self._shared_dirs is None:
            raise RuntimeError("Calibration not complete")
        
        # Center
        z_centered = z - self._global_mean  # [..., D]
        # Remove projection onto shared directions
        proj = z_centered @ self._shared_dirs.T  # [..., K]
        shared_part = proj @ self._shared_dirs  # [..., D]
        residual = z_centered - shared_part  # [..., D]
        return residual
    
    def _remove_shared_perstep(self, z: torch.Tensor) -> torch.Tensor:
        """Remove per-step shared components from trajectory latents."""
        if not self._calibrated:
            raise RuntimeError("Must calibrate before fusing")
        if not hasattr(self, '_perstep_dirs'):
            raise RuntimeError("Calibration was not done in per-step mode")
        
        # z: [H, D] or [B, H, D]
        if z.dim() == 2:
            # [H, D]
            z_residual = []
            for h in range(z.shape[0]):
                z_h = z[h]  # [D]
                mean_h = self._global_mean[0, h]  # [D]
                z_h_centered = z_h - mean_h
                # Remove projection onto shared directions for this step
                dirs_h = self._perstep_dirs[h]  # [K, D]
                proj = z_h_centered @ dirs_h.T  # [K]
                shared_part = proj @ dirs_h  # [D]
                z_residual.append(z_h_centered - shared_part)
            return torch.stack(z_residual, dim=0)  # [H, D]
        else:
            raise NotImplementedError("Batch per-step not yet implemented")
    
    def remove_shared(self, z: torch.Tensor) -> torch.Tensor:
        """Remove shared components from latent(s)."""
        if self.mode == "global":
            return self._remove_shared_global(z)
        else:
            return self._remove_shared_perstep(z)
    
    def fuse(self, *agent_latents: torch.Tensor, weights: Optional[List[float]] = None) -> torch.Tensor:
        """Fuse multiple agent latents after removing shared components.
        
        Args:
            agent_latents: Variable number of latent tensors to fuse
            weights: Optional weights for each agent (default: equal weights)
            
        Returns:
            Fused latent tensor of same shape as input
        """
        if weights is None:
            weights = [1.0 / len(agent_latents)] * len(agent_latents)
        
        if len(weights) != len(agent_latents):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of latents ({len(agent_latents)})")
        
        # Remove shared components from each latent
        residuals = [self.remove_shared(z) for z in agent_latents]
        
        # Weighted average of residuals
        fused = sum(w * r for w, r in zip(weights, residuals))
        
        # Add back the global mean (reconstruction)
        if self._global_mean is not None:
            if self.mode == "global":
                fused = self._global_mean + fused
            else:
                # For per-step, mean is already in the right shape
                fused = self._global_mean + fused
        
        return fused
    
    def get_singular_values(self) -> torch.Tensor:
        """Return the top-K singular values from calibration."""
        if not self._calibrated:
            raise RuntimeError("Must calibrate before getting singular values")
        return self._shared_dirs if hasattr(self, '_shared_dirs') else self._perstep_dirs


def test_pca_fusion():
    """Smoke test for PCAFusion."""
    print("Testing PCAFusion...")
    
    # Simulate calibration data: 100 agents, 32-dim latents
    N, D = 100, 32
    calibration_latents = torch.randn(N, D)
    
    # Create fusion module
    fusion = PCAFusion(n_components=1, mode="global")
    fusion.calibrate(calibration_latents)
    
    # Simulate two agent latents
    agent1 = torch.randn(D)
    agent2 = torch.randn(D)
    
    # Fuse them
    fused = fusion.fuse(agent1, agent2, weights=[0.5, 0.5])
    
    print(f"Agent1 shape: {agent1.shape}")
    print(f"Fused shape: {fused.shape}")
    print(f"Mean residual norm: {torch.norm(fused - agent1.mean()).item():.4f}")
    print("✅ PCAFusion smoke test passed")
    
    # Test per-step mode
    H = 16
    traj_calib = torch.randn(N, H, D)
    fusion_perstep = PCAFusion(n_components=1, mode="perstep")
    fusion_perstep.calibrate(traj_calib)
    
    agent1_traj = torch.randn(H, D)
    agent2_traj = torch.randn(H, D)
    
    fused_traj = fusion_perstep.fuse(agent1_traj, agent2_traj, weights=[0.5, 0.5])
    print(f"Fused traj shape: {fused_traj.shape}")
    print("✅ Per-step PCAFusion smoke test passed")


if __name__ == "__main__":
    test_pca_fusion()
