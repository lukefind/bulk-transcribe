"""
Compute backend detection and validation.

This module provides authoritative detection of available compute backends
based on the actual runtime environment. No assumptions, no silent fallbacks.
"""

import os
import platform
from typing import Dict, List, Any


def _is_in_docker() -> bool:
    """Detect if running inside a Docker container."""
    # Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True
    
    # Check cgroup for docker/container indicators
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'kubepods' in content or 'containerd' in content:
                return True
    except (FileNotFoundError, PermissionError):
        pass
    
    # Check for container environment variable
    if os.environ.get('DOCKER_CONTAINER') or os.environ.get('container'):
        return True
    
    return False


def _detect_cuda() -> bool:
    """Detect if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def _detect_metal() -> bool:
    """
    Detect if Metal (MPS) is available via PyTorch.
    
    Metal is ONLY available on:
    - macOS
    - Apple Silicon (arm64)
    - NOT inside Docker (Docker on macOS doesn't expose Metal)
    - PyTorch with MPS support
    """
    # Metal is never available in Docker
    if _is_in_docker():
        return False
    
    # Metal only on macOS
    if platform.system() != 'Darwin':
        return False
    
    # Metal only on arm64 (Apple Silicon)
    if platform.machine() not in ('arm64', 'aarch64'):
        return False
    
    # Check PyTorch MPS availability
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Additional check: can we actually use it?
            if torch.backends.mps.is_built():
                return True
    except ImportError:
        pass
    except Exception:
        pass
    
    return False


def detect_environment() -> Dict[str, Any]:
    """
    Detect the runtime environment and available compute backends.
    
    Returns:
        Dict with environment info and supported backends
    """
    system = platform.system().lower()
    if system == 'darwin':
        os_name = 'macos'
    elif system == 'linux':
        os_name = 'linux'
    else:
        os_name = system
    
    arch = platform.machine()
    if arch in ('AMD64', 'x86_64'):
        arch = 'x86_64'
    elif arch in ('arm64', 'aarch64'):
        arch = 'arm64'
    
    in_docker = _is_in_docker()
    cuda_available = _detect_cuda()
    metal_available = _detect_metal()
    
    # Build supported backends list
    supported_backends = ['cpu']  # CPU always available
    
    if cuda_available:
        supported_backends.append('cuda')
    
    if metal_available:
        supported_backends.append('metal')
    
    # Determine recommended backend
    if cuda_available:
        recommended = 'cuda'
    elif metal_available:
        recommended = 'metal'
    else:
        recommended = 'cpu'
    
    return {
        'os': os_name,
        'arch': arch,
        'inDocker': in_docker,
        'cudaAvailable': cuda_available,
        'metalAvailable': metal_available,
        'supportedBackends': supported_backends,
        'recommendedBackend': recommended
    }


def validate_backend(requested_backend: str) -> tuple:
    """
    Validate that a requested backend is supported.
    
    Args:
        requested_backend: The backend requested by the user
    
    Returns:
        Tuple of (is_valid: bool, error_dict: dict or None)
    """
    env = detect_environment()
    supported = env['supportedBackends']
    
    if requested_backend not in supported:
        return False, {
            'code': 'BACKEND_UNSUPPORTED',
            'message': f'{requested_backend.capitalize()} backend is not available in this environment. Available: {", ".join(supported)}.'
        }
    
    return True, None


def get_torch_device(backend: str) -> str:
    """
    Convert backend name to PyTorch device string.
    
    Args:
        backend: One of 'cpu', 'cuda', 'metal'
    
    Returns:
        PyTorch device string
    """
    if backend == 'cuda':
        return 'cuda'
    elif backend == 'metal':
        return 'mps'
    else:
        return 'cpu'


# Cache environment detection at module load for performance
_cached_environment = None


def get_cached_environment() -> Dict[str, Any]:
    """Get cached environment detection (computed once at startup)."""
    global _cached_environment
    if _cached_environment is None:
        _cached_environment = detect_environment()
    return _cached_environment


def refresh_environment_cache() -> Dict[str, Any]:
    """Force refresh of environment cache."""
    global _cached_environment
    _cached_environment = detect_environment()
    return _cached_environment
