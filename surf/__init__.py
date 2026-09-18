from surf.surf import SURF, SURF3d, ConeCME, Observer, load_SURF_run, _setup_dirs_

# ============================================================================
# Automatic Differentiation Support
# ============================================================================

# Import autodiff module if PyTorch is available
try:
    from surf.surf_autodiff import AutodiffSolver
    _AUTODIFF_AVAILABLE = True
except ImportError:
    _AUTODIFF_AVAILABLE = False
    AutodiffSolver = None

# Advanced features with automatic differentiation
try:
    from surf.surf_autodiff_particles import ParticleTracker
    from surf.surf_cme_injection import CMEInjector
    from surf.surf_ensemble import EnsembleSimulator, EnsembleDataset
    from surf.surf_data_assimilation import (
        FourDVar, EnsembleKalmanFilter, ParticleFilter, ObservationOperator
    )
    from surf.surf_cost_analysis import (
        CostAnalyzer, DataAssimilationCostAnalyzer, generate_cost_report
    )
    _ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    _ADVANCED_FEATURES_AVAILABLE = False

# JAX backend option (optional)
try:
    from surf.surf_autodiff_jax import JAXSolver, create_jax_loss_function
    _JAX_BACKEND_AVAILABLE = True
except ImportError:
    _JAX_BACKEND_AVAILABLE = False

# ============================================================================
# Export all public APIs
# ============================================================================

__all__ = [
    # Core SURF
    'SURF', 'SURF3d', 'ConeCME', 'Observer', 'load_SURF_run',
    # AutoDiff
    'AutodiffSolver',
    # Advanced Features
    'ParticleTracker',
    'CMEInjector',
    'EnsembleSimulator',
    'EnsembleDataset',
    'FourDVar',
    'EnsembleKalmanFilter',
    'ParticleFilter',
    'ObservationOperator',
    'CostAnalyzer',
    'DataAssimilationCostAnalyzer',
    'generate_cost_report',
    # JAX Backend
    'JAXSolver',
    'create_jax_loss_function',
    # Flags
    '_AUTODIFF_AVAILABLE',
    '_ADVANCED_FEATURES_AVAILABLE',
    '_JAX_BACKEND_AVAILABLE',
]
