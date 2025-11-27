import os
import time
import logging
import csv
import json
import glob
from typing import Tuple, Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter

# Set default tensor type and seeds for reproducibility
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################
# Configuration Management
###############################

@dataclass
class DomainConfig:
    """Domain configuration parameters"""
    z_min: float = 4.5
    z_max: float = 6.0
    x_min: float = 0.0
    x_max: float = 9.0
    time_min: float = 0.0
    time_max: float = 60.0
    pinn_time_offset: float = 307.0

@dataclass
class SoilParameters:
    """Van Genuchten soil parameters"""
    theta_r: float = 0.04
    theta_s: float = 0.4
    vg_alpha: float = 1.2
    n: float = 1.68
    K_s: float = 0.15
    l: float = 0.5
    RU_max: float = 0.00435
    
    @property
    def m(self) -> float:
        return 1 - 1 / self.n

@dataclass
class PetrophysicalParameters:
    """Parameters for ERT resistivity to saturation conversion"""
    cementation_exponent: float = 2.17
    saturation_exponent: float = 1.33
    surface_conductivity: float = 0.22
    water_conductivity: float = 1.0
    porosity: float = 0.40
    irreducible_water_content: float = 0.04

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    num_epochs: int = 1000
    num_cycles: int = 8
    resampling_sample: int = 20000
    resampling_new: int = 500
    learning_rate: float = 5e-4
    early_stop_patience: int = 500
    min_epochs_per_cycle: int = 20
    max_cycles_without_improvement: int = 3
    grad_clip_norm: float = 1.0
    ert_weight_ic: float = 80.0
    ert_weight_data: float = 50.0
    lambda_transpiration: float = 50.0  # Weight for transpiration loss
    transpiration_integration_points: int = 2000  # Points for spatial integration
    checkpoint_interval: int = 100
    batch_size: int = 1000
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "output"

@dataclass
class ModelConfig:
    """Neural network model configuration"""
    input_dim: int = 3
    hidden_dim: int = 64
    num_hidden_layers: int = 4

@dataclass
class Config:
    """Main configuration container"""
    domain: DomainConfig = field(default_factory=DomainConfig)
    soil: SoilParameters = field(default_factory=SoilParameters)
    petro: PetrophysicalParameters = field(default_factory=PetrophysicalParameters)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # File paths
    mesh_file: str = r"C:/Users/caners/Desktop/caner/core_samples_gilgal/Red/hydrus_output/Meshtria.txt"
    pressure_file: str = r"C:/Users/caners/Desktop/caner/core_samples_gilgal/Red/hydrus_output/H.txt"
    ert_data_dir: str = r"C:/Users/caners/Desktop/caner/core_samples_gilgal/Red/r2_inverse_modeling_hydrus/r2_forward_times"
    
    # Monitoring configuration
    monitoring_locations: List[Tuple[float, float]] = field(default_factory=lambda: [(6.0, 5.7), (5.0, 5.5), (4.5, 5.2)])
    monitoring_time_steps: List[float] = field(default_factory=lambda: [307, 308, 309, 310, 312, 314, 316, 318, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365])
    
    def get_pinn_to_hydrus_map(self) -> Dict[float, float]:
        """Get mapping from PINN time to Hydrus time"""
        return {
            0: 307, 1: 308, 2: 309, 3: 310, 5: 312, 7: 314, 9: 316, 11: 318,
            13: 320, 18: 325, 23: 330, 28: 335, 33: 340, 38: 345, 43: 350,
            48: 355, 53: 360, 58: 365
        }
    
    def save_json(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# Global configuration instance
config = Config()

###############################
# Enhanced Logging Setup
###############################

class TrainingMonitor:
    """Monitor training progress with TensorBoard and logging"""
    def __init__(self, config: Config):
        self.config = config
        self.writer = None
        if config.training.use_tensorboard:
            self.writer = SummaryWriter(config.training.tensorboard_dir)
        self.metrics = defaultdict(list)
        self.step = 0
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to tensorboard and internal storage"""
        if step is None:
            step = self.step
        
        for key, value in metrics.items():
            if self.writer:
                self.writer.add_scalar(key, value, step)
            self.metrics[key].append(value)
        
        self.step = step + 1
    
    def log_model_gradients(self, model: nn.Module, step: Optional[int] = None):
        """Log gradient statistics"""
        if not self.writer:
            return
            
        if step is None:
            step = self.step
            
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
        
        total_norm = total_norm ** 0.5
        self.writer.add_scalar('gradients/total_norm', total_norm, step)
    
    def close(self):
        """Close the writer"""
        if self.writer:
            self.writer.close()

def setup_logging(log_file: str = "pinn_training.log", level: int = logging.INFO) -> None:
    """Enhanced logging setup"""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)

###############################
# Checkpoint Manager
###############################

class CheckpointManager:
    """Manage model checkpoints for training recovery"""
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, cycle: int, loss: float, best_loss: float,
                       colloc_points: torch.Tensor, config: Config) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'cycle': cycle,
            'loss': loss,
            'best_loss': best_loss,
            'colloc_points': colloc_points.cpu(),
            'config': asdict(config),
            'timestamp': time.time()
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_cycle{cycle}_epoch{epoch}.pt')
        torch.save(checkpoint, path)
        logging.debug(f"Checkpoint saved: {path}")
        
        # Also save best model
        if loss <= best_loss:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved with loss: {loss:.4e}")
        
        # Keep only last 5 checkpoints
        self._cleanup_old_checkpoints(keep_last=5)
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the most recent checkpoint"""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.pt'))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=os.path.getctime)
        logging.info(f"Loading checkpoint: {latest}")
        return torch.load(latest, map_location=device)
    
    def load_best_checkpoint(self) -> Optional[Dict]:
        """Load the best model checkpoint"""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            logging.info(f"Loading best model from: {best_path}")
            return torch.load(best_path, map_location=device)
        return None
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.pt'))
        checkpoints.sort(key=os.path.getctime)
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                os.remove(checkpoint)
                logging.debug(f"Removed old checkpoint: {checkpoint}")

###############################
# Data Handling Classes
###############################

class HydrusDataHandler:
    """Enhanced Hydrus data handler with better error handling"""
    def __init__(self, config: Config):
        self.config = config
        self.nodes: Optional[Dict[int, Tuple[float, float]]] = None
        self.elements: Optional[List[List[int]]] = None
        self.domain_size: Optional[Dict[str, float]] = None
        self.time_steps: List[float] = []
        self.pressure_data: Optional[Dict[float, List[float]]] = None

    def load_mesh(self, file_content: str) -> Tuple[Dict[int, Tuple[float, float]], List[List[int]], Dict[str, float]]:
        """Load and parse Hydrus mesh with validation"""
        try:
            nodes, elements, domain_size = self.parse_hydrus_mesh(file_content)
            
            # Validate domain
            if domain_size["z_min"] > self.config.domain.z_min:
                logging.warning(f"Mesh z_min ({domain_size['z_min']}) > configured z_min ({self.config.domain.z_min})")
            
            # Override domain bounds to reduced domain
            domain_size["z_min"] = self.config.domain.z_min
            domain_size["z_max"] = self.config.domain.z_max
            
            logging.info(f"Mesh loaded: {len(nodes)} nodes, {len(elements)} elements")
            logging.info(f"Domain modified to reduced zone: z ∈ [{self.config.domain.z_min}, {self.config.domain.z_max}]")
            
            self.nodes = nodes
            self.elements = elements
            self.domain_size = domain_size
            
            return nodes, elements, domain_size
            
        except Exception as e:
            logging.error(f"Error loading mesh: {e}")
            raise RuntimeError(f"Failed to load mesh: {e}")

    def load_pressure(self, file_content: str) -> List[float]:
        """Load pressure data with validation"""
        try:
            lines = file_content.splitlines()
            self.time_steps = []
            self.pressure_data = {}
            current_time: Optional[float] = None
            current_values: List[float] = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("Time"):
                    if current_time is not None and current_values:
                        self.pressure_data[current_time] = current_values
                        current_values = []
                    try:
                        current_time = float(line.split('=')[1].strip())
                        self.time_steps.append(current_time)
                        logging.debug(f"Parsed time step: {current_time}")
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Could not parse time in line: {line} | Error: {e}")
                        continue
                else:
                    try:
                        values = [float(v) for v in line.split()]
                        current_values.extend(values)
                    except ValueError:
                        continue
            
            if current_time is not None and current_values:
                self.pressure_data[current_time] = current_values
            
            if not self.pressure_data:
                raise ValueError("No pressure data found in file")
            
            logging.info(f"Loaded pressure data for {len(self.time_steps)} time steps")
            return self.time_steps
            
        except Exception as e:
            logging.error(f"Error loading pressure data: {e}")
            raise RuntimeError(f"Failed to load pressure data: {e}")

    def parse_hydrus_mesh(self, file_content: str) -> Tuple[Dict[int, Tuple[float, float]], List[List[int]], Dict[str, float]]:
        """Parse Hydrus mesh file with enhanced error handling"""
        lines = file_content.splitlines()
        nodes: Dict[int, Tuple[float, float]] = {}
        elements: List[List[int]] = []
        domain_size: Dict[str, float] = {
            "x_min": float("inf"), "x_max": float("-inf"),
            "z_min": float("inf"), "z_max": float("-inf")
        }
        
        try:
            header = lines[0].split()
            n_nodes = int(header[1])
            n_elements = int(header[3])
            logging.info(f"Parsing mesh: {n_nodes} nodes, {n_elements} elements")
            
            idx = 1
            
            # Parse nodes
            while idx < len(lines) and len(nodes) < n_nodes:
                line = lines[idx].strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        z = float(parts[2])
                        nodes[node_id] = (x, z)
                        domain_size["x_min"] = min(domain_size["x_min"], x)
                        domain_size["x_max"] = max(domain_size["x_max"], x)
                        domain_size["z_min"] = min(domain_size["z_min"], z)
                        domain_size["z_max"] = max(domain_size["z_max"], z)
                idx += 1
            
            # Parse elements
            while idx < len(lines) and len(elements) < n_elements:
                line = lines[idx].strip()
                if line:
                    try:
                        parts = [int(p) for p in line.split()]
                        if len(parts) >= 4:
                            elements.append([parts[1] - 1, parts[2] - 1, parts[3] - 1])
                    except ValueError:
                        pass
                idx += 1
            
            if len(nodes) != n_nodes:
                logging.warning(f"Expected {n_nodes} nodes, parsed {len(nodes)}")
            if len(elements) != n_elements:
                logging.warning(f"Expected {n_elements} elements, parsed {len(elements)}")
            
            logging.info(f"Parsed {len(nodes)} nodes and {len(elements)} elements")
            
        except Exception as e:
            logging.error(f"Error parsing mesh file: {e}")
            raise e
        
        return nodes, elements, domain_size

    def interpolate_data(self, target_time: float, num_x: int = 100, num_z: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate pressure data at target time with domain filtering"""
        if self.pressure_data is None or self.nodes is None:
            raise ValueError("Pressure data or nodes not loaded.")
        
        if target_time not in self.pressure_data:
            closest_time = min(self.time_steps, key=lambda t: abs(t - target_time))
            logging.warning(f"Target time {target_time} not found; using closest time {closest_time}")
            target_time = closest_time
        
        # Use the reduced domain bounds
        x = np.linspace(self.config.domain.x_min, self.config.domain.x_max, num_x)
        z = np.linspace(self.config.domain.z_min, self.config.domain.z_max, num_z)
        X, Z = np.meshgrid(x, z)
        
        # Filter nodes to only include those within the reduced domain
        x_coords = []
        z_coords = []
        pressures = []
        
        for node_id, (node_x, node_z) in self.nodes.items():
            if self.config.domain.z_min <= node_z <= self.config.domain.z_max:
                x_coords.append(node_x)
                z_coords.append(node_z)
                if node_id - 1 < len(self.pressure_data[target_time]):
                    pressures.append(self.pressure_data[target_time][node_id - 1])
                else:
                    pressures.append(0.0)
        
        x_coords = np.array(x_coords)
        z_coords = np.array(z_coords)
        pressures = np.array(pressures)
        
        if len(x_coords) == 0:
            raise ValueError("No nodes found in the reduced domain")
        
        points = np.column_stack((x_coords, z_coords))
        grid_pressures = griddata(points, pressures, (X, Z), method="linear")
        
        # Handle NaN values
        nan_mask = np.isnan(grid_pressures)
        if np.any(nan_mask):
            logging.warning(f"Found {np.sum(nan_mask)} NaN values in interpolated data, filling with nearest")
            grid_pressures = griddata(points, pressures, (X, Z), method="nearest")
        
        logging.info(f"Interpolated data in reduced domain: {len(x_coords)} nodes used")
        return X, Z, grid_pressures

    def get_values_at_locations(self, locations: List[Tuple[float, float]], 
                               time_steps: Optional[List[float]] = None) -> Dict[float, List[float]]:
        """Get pressure values at specific locations and times"""
        if self.pressure_data is None or self.nodes is None:
            raise ValueError("Pressure data or nodes not loaded.")
        
        if time_steps is None:
            time_steps = self.time_steps
        else:
            available = set(self.time_steps)
            time_steps = [t if t in available else min(self.time_steps, key=lambda s: abs(s - t))
                         for t in time_steps]
        
        # Filter locations to be within the reduced domain
        filtered_locations = [(x, z) for x, z in locations 
                             if self.config.domain.z_min <= z <= self.config.domain.z_max]
        
        if len(filtered_locations) < len(locations):
            logging.warning(f"Some monitoring locations outside reduced domain. "
                          f"Using {len(filtered_locations)}/{len(locations)} locations.")
        
        # Filter nodes to reduced domain
        x_coords = []
        z_coords = []
        node_indices = []
        
        for node_id, (node_x, node_z) in self.nodes.items():
            if self.config.domain.z_min <= node_z <= self.config.domain.z_max:
                x_coords.append(node_x)
                z_coords.append(node_z)
                node_indices.append(node_id)
        
        x_coords = np.array(x_coords)
        z_coords = np.array(z_coords)
        
        values_dict: Dict[float, List[float]] = {}
        
        for t in time_steps:
            if t not in self.pressure_data:
                logging.warning(f"Time {t} not found in pressure data")
                continue
            
            # Filter pressures to reduced domain
            pressures = []
            for i, node_id in enumerate(node_indices):
                if node_id - 1 < len(self.pressure_data[t]):
                    pressures.append(self.pressure_data[t][node_id - 1])
                else:
                    pressures.append(0.0)
            
            pressures = np.array(pressures)
            points = np.column_stack((x_coords, z_coords))
            
            loc_values = []
            for (x, z) in filtered_locations:
                value = griddata(points, pressures, (x, z), method="linear")
                if np.isnan(value):
                    value = griddata(points, pressures, (x, z), method="nearest")
                loc_values.append(float(value))
            
            values_dict[t] = loc_values
        
        return values_dict

class ERTDataHandler:
    """Enhanced ERT data handler with better error handling and validation"""
    def __init__(self, config: Config):
        self.config = config
        self.base_dir = config.ert_data_dir
        self.timesteps = config.monitoring_time_steps
        self.ert_data = {}
        
    def read_vtk(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse VTK file with error handling"""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # Find POINTS section
            pts = None
            for i, L in enumerate(lines):
                if L.startswith('POINTS'):
                    n_pts = int(L.split()[1])
                    pts = []
                    for j in range(n_pts):
                        pts.append(list(map(float, lines[i+1+j].split())))
                    pts = np.array(pts)
                    break
            
            if pts is None:
                raise ValueError("No POINTS section found in VTK file")
            
            # Find CELLS section
            cells = None
            for i, L in enumerate(lines):
                if L.startswith('CELLS'):
                    n_cells = int(L.split()[1])
                    cells = []
                    for j in range(n_cells):
                        nums = list(map(int, lines[i+1+j].split()))
                        cnt = nums[0]
                        cells.append(nums[1:1+cnt])
                    cells = np.array(cells)
                    break
            
            if cells is None:
                raise ValueError("No CELLS section found in VTK file")
            
            # Find Resistivity scalars
            resistivity = None
            for i, L in enumerate(lines):
                if L.strip() == 'SCALARS Resistivity(ohm.m) double 1':
                    start = i + 2
                    vals = []
                    idx = start
                    while len(vals) < len(cells):
                        vals.extend(map(float, lines[idx].split()))
                        idx += 1
                    resistivity = np.array(vals[:len(cells)])
                    break
            
            if resistivity is None:
                raise ValueError("No Resistivity data found in VTK file")
            
            return pts, cells, resistivity
            
        except Exception as e:
            logging.error(f"Error reading VTK file {path}: {e}")
            raise
    
    def load_ert_data(self) -> Dict:
        """Load ERT data for all timesteps with validation"""
        loaded_count = 0
        failed_count = 0
        
        for t in self.timesteps:
            fn = os.path.join(self.base_dir, f'R2_forward_{t}.vtk')
            
            if not os.path.exists(fn):
                logging.warning(f"ERT file not found: {fn}")
                failed_count += 1
                continue
            
            try:
                pts, cells, resistivity = self.read_vtk(fn)
                
                # Calculate center of each cell
                cell_centers = np.array([pts[cell].mean(axis=0) for cell in cells])
                
                # Extract X and Z coordinates (X, Z plane)
                coords_original = cell_centers[:, [0, 2]]
                
                # Transform and filter coordinates
                valid_x_indices = (coords_original[:, 0] >= 0) & (coords_original[:, 0] <= 9)
                
                coords_transformed = coords_original[valid_x_indices].copy()
                coords_transformed[:, 1] = 6 + coords_original[valid_x_indices, 1]
                
                # Filter to reduced domain
                valid_z_indices = (coords_transformed[:, 1] >= self.config.domain.z_min) & \
                                 (coords_transformed[:, 1] <= self.config.domain.z_max)
                coords_final = coords_transformed[valid_z_indices]
                
                # Filter resistivity values
                resistivity_filtered = resistivity[valid_x_indices][valid_z_indices]
                
                # Validate data
                if len(coords_final) == 0:
                    logging.warning(f"No valid ERT data points for timestep {t}")
                    failed_count += 1
                    continue
                
                self.ert_data[t] = {
                    'coords': coords_final,
                    'resistivity': resistivity_filtered
                }
                
                loaded_count += 1
                logging.info(f"Loaded ERT data for timestep {t}: {len(coords_final)} cells "
                           f"in reduced domain z ∈ [{self.config.domain.z_min}, {self.config.domain.z_max}]")
                
            except Exception as e:
                logging.error(f"Failed to load ERT data for timestep {t}: {e}")
                failed_count += 1
        
        logging.info(f"ERT data loading complete: {loaded_count} successful, {failed_count} failed")
        
        if loaded_count == 0:
            raise RuntimeError("No ERT data could be loaded successfully")
        
        return self.ert_data
    
    def resistivity_to_saturation(self, resistivity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert resistivity to water saturation using petrophysical relationship"""
        # Handle arrays
        if isinstance(resistivity, np.ndarray):
            return np.array([self.resistivity_to_saturation(r) for r in resistivity])
        
        # Handle extreme values
        if resistivity > 1e8:
            return 0.01
        if resistivity < 1e-8:
            return 1.0
        
        # Convert resistivity to conductivity
        sigma_b = 1.0 / resistivity
        
        # Formation factor
        formation_factor = self.config.petro.porosity ** self.config.petro.cementation_exponent
        
        # Newton-Raphson iteration to solve for Sw
        def func(Sw):
            return formation_factor * (Sw ** self.config.petro.saturation_exponent) * \
                   (self.config.petro.water_conductivity + self.config.petro.surface_conductivity / Sw) - sigma_b
        
        def dfunc(Sw):
            return formation_factor * (
                self.config.petro.saturation_exponent * (Sw ** (self.config.petro.saturation_exponent - 1)) * 
                (self.config.petro.water_conductivity + self.config.petro.surface_conductivity / Sw) - 
                (Sw ** self.config.petro.saturation_exponent) * self.config.petro.surface_conductivity / (Sw ** 2)
            )
        
        # Initial guess
        Sw = 0.5
        
        # Newton-Raphson iteration
        max_iter = 50
        tol = 1e-6
        
        for _ in range(max_iter):
            f = func(Sw)
            if abs(f) < tol:
                break
            
            df = dfunc(Sw)
            if abs(df) < 1e-10:  # Avoid division by zero
                break
            
            Sw_new = Sw - f / df
            
            # Ensure Sw stays within bounds
            Sw_new = np.clip(Sw_new, 0.01, 1.0)
            
            if abs(Sw_new - Sw) < tol:
                break
            
            Sw = Sw_new
        
        return np.clip(Sw, 0.01, 1.0)
    
    

    def convert_to_saturation(self) -> Dict:
        """Convert all resistivity data to *effective* saturation Se."""
        # Decide S_wr source:
        # Priority 1: explicit petro.irreducible_water_content, else from theta_r/porosity
        if getattr(self.config.petro, "irreducible_water_content", None) is not None:
            S_wr = float(self.config.petro.irreducible_water_content)
        else:
            S_wr = self.config.soil.theta_r / max(self.config.petro.porosity, 1e-6)
    
        for t in self.ert_data:
            resistivity = self.ert_data[t]['resistivity']
    
            # 1) Resistivity -> conductivity -> invert Waxman–Smits -> Sw
            Sw = self.resistivity_to_saturation(resistivity)  # returns Sw
    
            # 2) Sw -> Se (this is the key change)
            Se = Sw_to_Se_numpy(Sw, S_wr)
    
            # 3) Store Se instead of Sw
            self.ert_data[t]['saturation'] = Se
    
            # Logging
            min_se = float(np.min(Se)); max_se = float(np.max(Se)); mean_se = float(np.mean(Se))
            logging.info(
                f"Converted ERT at t={t}: S_wr={S_wr:.4f}, "
                f"Se range [{min_se:.3f}, {max_se:.3f}], mean={mean_se:.3f}"
            )
    
        return self.ert_data

    
    def get_initial_condition_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get ERT data at initial time for use as initial condition"""
        initial_time = self.config.monitoring_time_steps[0]
        
        if initial_time not in self.ert_data:
            raise ValueError(f"Initial time step (t={initial_time}) not found in ERT data")
        
        coords = self.ert_data[initial_time]['coords']
        saturation = self.ert_data[initial_time]['saturation']
        
        logging.info(f"Using ERT data at t={initial_time} as initial condition: "
                   f"{len(coords)} points in reduced domain")
        
        return coords, saturation


class HydrusUptakeDataHandler:
    """Handler for Hydrus root water uptake data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.uptake_data_dir = r"C:\Users\caners\Desktop\caner\core_samples_gilgal\Red\root_water_uptake_03_RWU_output"
        self.uptake_data = {}  # {timestep: {'coords': array, 'uptake': array, 'total_transpiration': float}}
        
    def load_uptake_data(self) -> Dict:
        """Load all available Hydrus uptake files"""
        import glob
        pattern = os.path.join(self.uptake_data_dir, "root_water_uptake_03_Root Water Uptake_root_water_uptake_*.txt")
        files = glob.glob(pattern)
        
        loaded_count = 0
        failed_count = 0
        
        for filepath in files:
            try:
                # Extract timestep from filename
                filename = os.path.basename(filepath)
                timestep_str = filename.split('_')[-1].replace('.txt', '')
                timestep = float(timestep_str)
                
                # Load the file
                coords, uptake = self._parse_uptake_file(filepath)
                
                # Filter to reduced domain (z between z_min and z_max)
                z_coords = coords[:, 1]
                valid_mask = (z_coords >= self.config.domain.z_min) & (z_coords <= self.config.domain.z_max)
                
                if np.any(valid_mask):
                    filtered_coords = coords[valid_mask]
                    filtered_uptake = uptake[valid_mask]
                    
                    # CONSISTENT INTEGRATION: Use mean * area like PINN
                    domain_area = (self.config.domain.x_max - self.config.domain.x_min) * \
                                 (self.config.domain.z_max - self.config.domain.z_min)
                    total_transpiration = np.mean(filtered_uptake) * domain_area
                    
                    self.uptake_data[timestep] = {
                        'coords': filtered_coords,
                        'uptake': filtered_uptake,
                        'total_transpiration': total_transpiration,
                        'raw_sum': np.sum(filtered_uptake),  # Keep original sum for debugging
                        'n_points': len(filtered_coords)
                    }
                    loaded_count += 1
                    
                    logging.info(f"Loaded Hydrus uptake data for timestep {timestep}: "
                               f"{np.sum(valid_mask)}/{len(coords)} points, "
                               f"mean uptake = {np.mean(filtered_uptake):.8f}, "
                               f"total transpiration = {total_transpiration:.6f}")
                else:
                    logging.warning(f"No uptake data points in reduced domain for timestep {timestep}")
                    failed_count += 1
                    
            except Exception as e:
                logging.error(f"Failed to load uptake file {filepath}: {e}")
                failed_count += 1
        
        logging.info(f"Hydrus uptake data loading complete: {loaded_count} successful, {failed_count} failed")
        
        if loaded_count == 0:
            raise RuntimeError("No Hydrus uptake data could be loaded successfully")
        
        return self.uptake_data
    
    def _parse_uptake_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a single Hydrus uptake file"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find the start of TABLE_01
        data_start = None
        for i, line in enumerate(lines):
            if 'TABLE_01' in line:
                # Skip header lines to find actual data
                for j in range(i, len(lines)):
                    if lines[j].startswith(';'):
                        continue
                    elif lines[j].strip() and not lines[j].startswith(';'):
                        data_start = j
                        break
                break
        
        if data_start is None:
            raise ValueError("Could not find TABLE_01 data in file")
        
        # Parse data
        coords = []
        uptake = []
        
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            try:
                parts = line.split()
                if len(parts) >= 4:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    z = float(parts[2])
                    u = float(parts[3])
                    
                    coords.append([x, z])
                    uptake.append(u)
            except (ValueError, IndexError):
                continue
        
        coords = np.array(coords)
        uptake = np.array(uptake)
        
        logging.debug(f"Parsed {len(coords)} uptake points from {os.path.basename(filepath)}")
        return coords, uptake
    
    def get_total_transpiration(self, timestep: float) -> float:
        """Get total transpiration for a specific timestep"""
        if timestep in self.uptake_data:
            return self.uptake_data[timestep]['total_transpiration']
        else:
            return 0.0

###############################
# Physics Functions
###############################

def h_to_S(h: Union[torch.Tensor, np.ndarray], config: Config) -> Union[torch.Tensor, np.ndarray]:
    """Convert pressure head to effective saturation using van Genuchten"""
    if isinstance(h, torch.Tensor):
        h_abs = torch.abs(h)
        return torch.where(h >= 0, torch.ones_like(h), 
                          torch.pow(1 + torch.pow(config.soil.vg_alpha * h_abs, config.soil.n), -config.soil.m))
    else:
        h_abs = np.abs(h)
        return np.where(h >= 0, np.ones_like(h), 
                       np.power(1 + np.power(config.soil.vg_alpha * h_abs, config.soil.n), -config.soil.m))

def S_to_h(S: Union[torch.Tensor, np.ndarray], config: Config) -> Union[torch.Tensor, np.ndarray]:
    """Convert effective saturation to pressure head using van Genuchten"""
    if isinstance(S, torch.Tensor):
        S_clipped = torch.clamp(S, 1e-6, 1.0)  # ✅ NEW: Updated to match scaled sigmoid
        return torch.where(S >= 1.0, torch.zeros_like(S),
                          -1 / config.soil.vg_alpha * torch.pow(torch.pow(S_clipped, -1/config.soil.m) - 1, 1/config.soil.n))
    else:
        S_clipped = np.clip(S, 1e-8, 1.0)
        return np.where(S >= 1.0, np.zeros_like(S),
                       -1 / config.soil.vg_alpha * np.power(np.power(S_clipped, -1/config.soil.m) - 1, 1/config.soil.n))


def Sw_to_Se_numpy(Sw: np.ndarray, S_wr: float) -> np.ndarray:
    """Convert water saturation Sw to effective saturation Se."""
    denom = max(1.0 - S_wr, 1e-6)
    Se = (Sw - S_wr) / denom
    return np.clip(Se, 0.0, 1.0)

def feddes(h: torch.Tensor) -> torch.Tensor:
    """Feddes root water uptake reduction factor (piecewise linear)"""
    h_wet, h_opt, h_dry, h_wilt = -0.1, -0.25, -5.0, -80.0
    
    f = torch.zeros_like(h)
    f = torch.where((h <= h_wet) & (h > h_opt), (h_wet - h) / (h_wet - h_opt), f)
    f = torch.where((h <= h_opt) & (h >= h_dry), torch.ones_like(h), f)
    f = torch.where((h < h_dry) & (h >= h_wilt), (h - h_wilt) / (h_dry - h_wilt), f)
    
    return f

def pde_residual(model: nn.Module, points: torch.Tensor, config: Config) -> torch.Tensor:
    """Compute the PDE residual for Richards' equation with root water uptake"""
    points.requires_grad_(True)
    S, U = model(points)
    
    # Compute gradients of saturation
    grad_S = torch.autograd.grad(S, points, grad_outputs=torch.ones_like(S),
                                create_graph=True, retain_graph=True)[0]
    S_t = grad_S[:, 2:3]
    theta_t = (config.soil.theta_s - config.soil.theta_r) * S_t
    
    # Convert to pressure head and compute hydraulic conductivity
    h = S_to_h(S, config)
    K = config.soil.K_s * (S ** config.soil.l) * \
        (1 - (1 - S ** (1/config.soil.m)) ** config.soil.m) ** 2
    
    # Compute gradients of pressure head
    grad_h = torch.autograd.grad(h, points, grad_outputs=torch.ones_like(h),
                                create_graph=True, retain_graph=True)[0]
    h_x, h_z = grad_h[:, 0:1], grad_h[:, 1:2]
    
    # Compute fluxes
    q_x = -K * h_x
    q_z = -K * (h_z + 1)
    
    # Compute divergence of fluxes
    grad_qx = torch.autograd.grad(q_x, points, grad_outputs=torch.ones_like(q_x),
                                 create_graph=True, retain_graph=True)[0]
    grad_qz = torch.autograd.grad(q_z, points, grad_outputs=torch.ones_like(q_z),
                                 create_graph=True, retain_graph=True)[0]
    q_x_x, q_z_z = grad_qx[:, 0:1], grad_qz[:, 1:2]
    
    # Since entire domain is root zone, uptake applies everywhere
    uptake = U
    
    # Richards equation residual
    residual = theta_t + (q_x_x + q_z_z) + uptake
    
    return residual

###############################
# Neural Network Models
###############################

class PINN(nn.Module):
    """Physics-Informed Neural Network for water saturation prediction"""
    def __init__(self, config: Config):
        super(PINN, self).__init__()
        self.config = config
        
        # Scaling factors
        self.x_scale = config.domain.x_max - config.domain.x_min
        self.z_scale = config.domain.z_max - config.domain.z_min
        self.t_scale = config.domain.time_max
        
        # Network architecture
        self.input_layer = nn.Linear(config.model.input_dim, config.model.hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim) 
            for _ in range(config.model.num_hidden_layers)
        ])
        self.output_layer = nn.Linear(config.model.hidden_dim, 1)
        
        # Activation functions
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # ✅ NEW: Saturation range parameters for scaled sigmoid
        self.s_min = 1e-6  # Much lower minimum (allows h down to ~-100m)
        self.s_max = 1.0   # Standard maximum
        
        # Initialize weights
        self._initialize_weights()
    
    # ✅ NEW: Add this method
    def scaled_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Scaled sigmoid to allow much lower saturation values"""
        return self.s_min + (self.s_max - self.s_min) * self.sigmoid(x)
    
    def _initialize_weights(self):
        """Xavier initialization for weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with input scaling and residual connections"""
        # Scale inputs
        x_scaled = x.clone()
        x_scaled[:, 0] /= self.x_scale
        x_scaled[:, 1] /= self.z_scale
        x_scaled[:, 2] /= self.t_scale
        
        # Forward pass
        out = self.activation(self.input_layer(x_scaled))
        
        # Hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            residual = out
            out = self.activation(layer(out))
            if i % 2 == 1:  # Add residual every other layer
                out = out + residual
        
        # ✅ NEW: Output layer with SCALED sigmoid for broader range
        return self.scaled_sigmoid(self.output_layer(out))

class PINNWithUptake(nn.Module):
    """Two-headed PINN for saturation and uptake prediction"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Saturation network
        self.pinn_net = PINN(config)
        
        # Uptake network (separate architecture)
        self.uptake_x_scale = config.domain.x_max - config.domain.x_min
        self.uptake_z_scale = config.domain.z_max - config.domain.z_min
        self.uptake_t_scale = config.domain.time_max
        
        # Build uptake network WITHOUT Softplus
        layers = [nn.Linear(config.model.input_dim, config.model.hidden_dim), nn.Tanh()]
        for _ in range(config.model.num_hidden_layers):
            layers += [nn.Linear(config.model.hidden_dim, config.model.hidden_dim), nn.Tanh()]
        layers += [nn.Linear(config.model.hidden_dim, 1)]  # No activation here
        
        self.uptake_net = nn.Sequential(*layers)
        
        # Initialize uptake network
        self._initialize_uptake_weights()
    
    def _initialize_uptake_weights(self):
        """Initialize uptake network weights"""
        for m in self.uptake_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both saturation and uptake"""
        # Saturation prediction
        S = self.pinn_net(x)
        
        # Scale inputs for uptake network
        x_up = x.clone()
        x_up[:, 0] /= self.uptake_x_scale
        x_up[:, 1] /= self.uptake_z_scale
        x_up[:, 2] /= self.uptake_t_scale
        
        # Uptake prediction with proper constraint
        U_raw = self.uptake_net(x_up)
        
        # Fixed constraint implementation
        # Maps to [0, RU_max] with smooth gradient
        U = self.config.soil.RU_max * torch.sigmoid(U_raw)
        
        return S, U

class HydrusInterpolator:
    """Bilinear interpolator for Hydrus saturation data"""
    def __init__(self, hydrus_data: Dict, config: Config):
        self.config = config
        self.X = hydrus_data["X"]
        self.Z = hydrus_data["Z"]
        self.S = hydrus_data["saturation"].reshape(self.X.shape)
        self.domain = hydrus_data["domain"]
        
        self.x_min = self.domain["x_min"]
        self.x_max = self.domain["x_max"]
        self.z_min = self.domain["z_min"]
        self.z_max = self.domain["z_max"]
        
        self.nx = self.X.shape[1]
        self.nz = self.X.shape[0]
        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        self.dz = (self.z_max - self.z_min) / (self.nz - 1)
        
        self.S_tensor = torch.tensor(self.S, dtype=torch.float32, device=device)
        
        logging.info(f"HydrusInterpolator initialized with grid {self.nx}x{self.nz}")
    
    def interpolate(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Perform bilinear interpolation"""
        # Normalize coordinates
        x_norm = (x - self.x_min) / self.dx
        z_norm = (z - self.z_min) / self.dz
        
        # Clamp to valid range
        x_norm = torch.clamp(x_norm, 0, self.nx - 1.001)
        z_norm = torch.clamp(z_norm, 0, self.nz - 1.001)
        
        # Get integer indices
        x0 = torch.floor(x_norm).long()
        z0 = torch.floor(z_norm).long()
        x1 = torch.clamp(x0 + 1, 0, self.nx - 1)
        z1 = torch.clamp(z0 + 1, 0, self.nz - 1)
        
        # Get fractional parts
        wx = x_norm - x0.float()
        wz = z_norm - z0.float()
        
        # Get corner values
        v00 = self.S_tensor[z0, x0]
        v01 = self.S_tensor[z0, x1]
        v10 = self.S_tensor[z1, x0]
        v11 = self.S_tensor[z1, x1]
        
        # Bilinear interpolation
        interpolated = (1 - wx) * (1 - wz) * v00 + \
                      wx * (1 - wz) * v01 + \
                      (1 - wx) * wz * v10 + \
                      wx * wz * v11
        
        return interpolated.reshape(-1, 1)

###############################
# Loss Functions
###############################

def ert_initial_condition_loss(model: nn.Module, ic_points: torch.Tensor, 
                              target_saturations: torch.Tensor) -> torch.Tensor:
    """Compute the initial condition loss using ERT-derived saturation"""
    ic_points.requires_grad_(True)
    S_pred, _ = model(ic_points)
    loss_ic = torch.mean((S_pred - target_saturations) ** 2)
    return loss_ic

def ert_data_loss(model: nn.Module, ert_points: torch.Tensor, 
                 target_saturations: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error loss between PINN predictions and ERT saturations"""
    S_pred, _ = model(ert_points)
    return torch.mean((S_pred - target_saturations) ** 2)

def measurement_loss(model: nn.Module, meas_points: torch.Tensor, 
                    target_vals: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss for measurement data"""
    S_pred, _ = model(meas_points)
    return torch.mean((S_pred - target_vals) ** 2)

def compute_pinn_total_transpiration(model: nn.Module, t: float, config: Config, 
                                   n_points: int = 2000) -> torch.Tensor:
    """Compute total transpiration from PINN by spatial integration"""
    # Generate spatial integration points using stratified sampling for better coverage
    # Divide domain into grid and sample within each cell
    n_x = int(np.sqrt(n_points * (config.domain.x_max - config.domain.x_min) / 
                     (config.domain.z_max - config.domain.z_min)))
    n_z = n_points // n_x
    
    x_grid = torch.linspace(config.domain.x_min, config.domain.x_max, n_x, device=device)
    z_grid = torch.linspace(config.domain.z_min, config.domain.z_max, n_z, device=device)
    
    # Create mesh grid
    X_mesh, Z_mesh = torch.meshgrid(x_grid, z_grid, indexing='ij')
    x_pts = X_mesh.flatten().unsqueeze(1)
    z_pts = Z_mesh.flatten().unsqueeze(1)
    
    actual_n_points = len(x_pts)
    t_pts = torch.full((actual_n_points, 1), t, device=device)
    
    integration_points = torch.cat([x_pts, z_pts, t_pts], dim=1)
    
    # Get uptake predictions
    _, U_pred = model(integration_points)
    
    # Integrate over domain (using trapezoidal rule approximation)
    domain_area = (config.domain.x_max - config.domain.x_min) * (config.domain.z_max - config.domain.z_min)
    total_transpiration = torch.mean(U_pred) * domain_area
    
    return total_transpiration

def transpiration_loss(model: nn.Module, hydrus_uptake_handler: HydrusUptakeDataHandler, 
                      t: float, config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute transpiration loss between PINN and Hydrus"""
    # Get PINN total transpiration
    pinn_total = compute_pinn_total_transpiration(model, t, config, 
                                                config.training.transpiration_integration_points)
    
    # Get Hydrus total transpiration
    hydrus_time = t + config.domain.pinn_time_offset
    hydrus_total = hydrus_uptake_handler.get_total_transpiration(hydrus_time)
    
    if hydrus_total == 0.0:
        # If no Hydrus data, return zero loss
        return torch.tensor(0.0, device=device), pinn_total, torch.tensor(0.0, device=device)
    
    hydrus_total_tensor = torch.tensor(hydrus_total, dtype=torch.float32, device=device)
    
    # Compute relative loss to handle different scales
    # Use relative difference to avoid issues with absolute magnitudes
    if hydrus_total > 1e-8:
        relative_error = (pinn_total - hydrus_total_tensor) / hydrus_total_tensor
        loss = relative_error ** 2
    else:
        # If Hydrus total is very small, use absolute difference
        loss = (pinn_total - hydrus_total_tensor) ** 2
    
    return loss, pinn_total, hydrus_total_tensor

def calc_boundary_loss(model: nn.Module, bc_points: torch.Tensor, 
                      bc_type: str, config: Config) -> torch.Tensor:
    """Compute the boundary condition loss"""
    bc_points.requires_grad_(True)
    S_val, _ = model(bc_points)
    h = S_to_h(S_val, config)
    K_val = config.soil.K_s * (S_val ** config.soil.l) * \
            (1 - (1 - S_val ** (1/config.soil.m)) ** config.soil.m) ** 2
    
    grad_h = torch.autograd.grad(h, bc_points, grad_outputs=torch.ones_like(h),
                                create_graph=True, retain_graph=True)[0]
    
    if bc_type == 'top':
        h_z = grad_h[:, 1:2]
        loss = torch.mean((-K_val * (h_z + 1)) ** 2)  # No-flux top boundary
    elif bc_type in ['left', 'right']:
        h_x = grad_h[:, 0:1]
        loss = torch.mean((-K_val * h_x) ** 2)  # No-flux lateral boundaries
    else:
        raise ValueError(f"Unknown boundary type: {bc_type}")
    
    return loss

###############################
# Data Generation and Sampling
###############################

class BatchedDataProcessor:
    """Handle large datasets efficiently with batching"""
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    def process_in_batches(self, data: np.ndarray, model: nn.Module, 
                          process_fn, device: torch.device) -> np.ndarray:
        """Process data in batches to avoid memory issues"""
        n_samples = len(data)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        results = []
        
        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_samples)
            
            batch = data[start:end]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                result = process_fn(model, batch_tensor)
            
            results.append(result.cpu().numpy())
            
            # Clear GPU cache periodically
            if device.type == "cuda" and i % 10 == 0:
                torch.cuda.empty_cache()
        
        return np.concatenate(results)

def generate_collocation_points(config: Config, N_colloc: int = 20000, 
                               N_bc_vert: int = 2000, N_bc_lat: int = 4000) -> Tuple[torch.Tensor, ...]:
    """Generate collocation points for interior and boundary conditions"""
    x_min, x_max = config.domain.x_min, config.domain.x_max
    z_min, z_max = config.domain.z_min, config.domain.z_max
    time_min, time_max = config.domain.time_min, config.domain.time_max
    
    # Interior collocation points with log-distributed time
    x_colloc = np.random.uniform(x_min, x_max, (N_colloc, 1))
    z_colloc = np.random.uniform(z_min, z_max, (N_colloc, 1))
    
    # Log-distributed time sampling for better coverage
    t_rand = np.random.uniform(0, 1, (N_colloc, 1))
    t_colloc = time_max * (np.exp(t_rand * np.log(1.1)) - 1) / 0.1
    
    colloc_points = torch.tensor(np.hstack((x_colloc, z_colloc, t_colloc)),
                                dtype=torch.float32, device=device, requires_grad=True)
    
    # Boundary points (top, left, right) - NO BOTTOM
    # Top boundary
    x_top = np.random.uniform(x_min, x_max, (N_bc_vert, 1))
    t_top = np.random.uniform(time_min, time_max, (N_bc_vert, 1))
    top_points = torch.tensor(np.hstack((x_top, z_max * np.ones((N_bc_vert, 1)), t_top)),
                             dtype=torch.float32, device=device, requires_grad=True)
    
    # Left boundary
    z_left = np.random.uniform(z_min, z_max, (N_bc_lat, 1))
    t_left = np.random.uniform(time_min, time_max, (N_bc_lat, 1))
    left_points = torch.tensor(np.hstack((x_min * np.ones((N_bc_lat, 1)), z_left, t_left)),
                              dtype=torch.float32, device=device, requires_grad=True)
    
    # Right boundary
    z_right = np.random.uniform(z_min, z_max, (N_bc_lat, 1))
    t_right = np.random.uniform(time_min, time_max, (N_bc_lat, 1))
    right_points = torch.tensor(np.hstack((x_max * np.ones((N_bc_lat, 1)), z_right, t_right)),
                               dtype=torch.float32, device=device, requires_grad=True)
    
    logging.info(f"Generated collocation points for reduced domain z ∈ [{z_min}, {z_max}]")
    logging.info(f"  Interior: {N_colloc}, Top: {N_bc_vert}, Left/Right: {N_bc_lat} each")
    
    return colloc_points, top_points, left_points, right_points

def prepare_ert_data_for_training(ert_data: Dict, config: Config) -> Tuple[Dict[float, torch.Tensor], Dict[float, torch.Tensor]]:
    """Prepare ERT data for training by converting to PyTorch tensors"""
    ert_points = {}
    ert_saturations = {}
    
    # Create reverse mapping from Hydrus time to PINN time
    pinn_to_hydrus_map = config.get_pinn_to_hydrus_map()
    hydrus_to_pinn_map = {v: k for k, v in pinn_to_hydrus_map.items()}
    
    for hydrus_t, data in ert_data.items():
        if hydrus_t in hydrus_to_pinn_map:
            pinn_t = hydrus_to_pinn_map[hydrus_t]
            coords = data['coords']
            n_points = len(coords)
            
            # Create points tensor with time
            points = np.hstack([
                coords,  # x, z coordinates
                np.full((n_points, 1), pinn_t)  # add time dimension
            ])
            
            # Convert to tensors
            ert_points[pinn_t] = torch.tensor(points, dtype=torch.float32, device=device)
            ert_saturations[pinn_t] = torch.tensor(data['saturation'].reshape(-1, 1), 
                                                 dtype=torch.float32, device=device)
            
            logging.info(f"Prepared ERT data for PINN time {pinn_t}: {n_points} points")
    
    return ert_points, ert_saturations

def generate_measurement_points(monitoring_locations: List[Tuple[float, float]],
                               time_steps: List[float],
                               pressure_values: Dict[float, List[float]],
                               config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate measurement points and target saturation values"""
    # Filter locations to reduced domain
    filtered_locations = [(x, z) for x, z in monitoring_locations 
                         if config.domain.z_min <= z <= config.domain.z_max]
    
    if len(filtered_locations) < len(monitoring_locations):
        logging.warning(f"Filtered monitoring locations to reduced domain: "
                       f"{len(filtered_locations)}/{len(monitoring_locations)} locations")
    
    n_locations = len(filtered_locations)
    n_points = n_locations * len(time_steps)
    points = np.zeros((n_points, 3))
    target_pressures = np.zeros((n_points, 1))
    
    idx = 0
    for t in time_steps:
        if t not in pressure_values:
            logging.warning(f"Time {t} not in pressure values, skipping")
            continue
        
        for i, (x, z) in enumerate(filtered_locations):
            points[idx, 0] = x
            points[idx, 1] = z
            points[idx, 2] = t - time_steps[0]  # normalize time
            
            if i < len(pressure_values[t]):
                target_pressures[idx, 0] = pressure_values[t][i]
            else:
                target_pressures[idx, 0] = 0.0
            idx += 1
    
    measurement_points = torch.tensor(points[:idx], dtype=torch.float32, device=device)
    target_sat = h_to_S(target_pressures[:idx], config)
    target_values = torch.tensor(target_sat, dtype=torch.float32, device=device)
    
    logging.info(f"Generated {idx} measurement points from {n_locations} locations")
    
    return measurement_points, target_values

def compute_residuals_and_resample(model: nn.Module, hydrus_handler: HydrusDataHandler,
                                  ert_handler: ERTDataHandler, config: Config,
                                  N_sample: int = 10000, N_new: int = 2000) -> torch.Tensor:
    """Generate new collocation points in high-error regions"""
    logging.info("Generating adaptive resampling points...")
    
    x_min, x_max = config.domain.x_min, config.domain.x_max
    z_min, z_max = config.domain.z_min, config.domain.z_max
    time_max = config.domain.time_max
    
    # Generate sample points
    x_sample = np.random.uniform(x_min, x_max, (N_sample, 1))
    z_sample = np.random.uniform(z_min, z_max, (N_sample, 1))
    t_rand = np.random.uniform(0, 1, (N_sample, 1))
    t_sample = time_max * (np.exp(t_rand * np.log(1.1)) - 1) / 0.1
    
    error_accum = np.ones(N_sample)
    
    # Compare with ERT data
    pinn_to_hydrus_map = config.get_pinn_to_hydrus_map()
    hydrus_to_pinn_map = {v: k for k, v in pinn_to_hydrus_map.items()}
    
    if ert_handler and ert_handler.ert_data:
        weight = 1.0
        
        # Sample 3 timesteps for comparison (prioritize early times)
        hydrus_times = sorted(ert_handler.ert_data.keys())[:3]
        
        for hydrus_t in hydrus_times:
            if hydrus_t not in hydrus_to_pinn_map:
                continue
            
            pinn_t = hydrus_to_pinn_map[hydrus_t]
            
            try:
                # Get ERT data
                coords = ert_handler.ert_data[hydrus_t]['coords']
                ert_saturations = ert_handler.ert_data[hydrus_t]['saturation']
                
                # Interpolate ERT saturations to sample points
                sample_points_2d = np.column_stack((x_sample, z_sample))
                S_ert = griddata(coords, ert_saturations, sample_points_2d, method="linear")
                
                # Get PINN predictions
                t_array = np.ones_like(x_sample) * pinn_t
                points_tensor = torch.tensor(np.hstack((x_sample, z_sample, t_array)),
                                           dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    S_pinn, _ = model(points_tensor)
                S_pinn = S_pinn.cpu().numpy()
                
                # Compute error
                valid_mask = ~np.isnan(S_ert)
                error = np.zeros_like(S_ert)
                error[valid_mask] = np.abs(S_pinn[valid_mask].flatten() - S_ert[valid_mask].flatten())
                
                error_accum += weight * error.flatten()
                weight *= 0.5
                
            except Exception as e:
                logging.error(f"Error during ERT comparison at time {hydrus_t}: {e}")
    
    # Compute PDE residual errors in batches WITHOUT torch.no_grad()
    pde_errors = np.zeros(N_sample)
    batch_size = config.training.batch_size
    n_batches = (N_sample + batch_size - 1) // batch_size
    
    logging.info(f"Computing PDE residuals in {n_batches} batches...")
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, N_sample)
        
        batch_points = torch.tensor(
            np.hstack((x_sample[start:end], z_sample[start:end], t_sample[start:end])),
            dtype=torch.float32, device=device, requires_grad=True
        )
        
        try:
            # Compute residual WITH gradients enabled
            residual = pde_residual(model, batch_points, config)
            batch_error = torch.abs(residual).detach().cpu().numpy().flatten()
            pde_errors[start:end] = batch_error
            
        except Exception as e:
            logging.error(f"PDE residual error in batch {i}: {e}")
            # Fill with zeros if error occurs
            pde_errors[start:end] = 0.0
        
        # Clear GPU cache periodically
        if device.type == "cuda" and i % 10 == 0:
            torch.cuda.empty_cache()
    
    # Normalize and combine errors
    if np.max(pde_errors) > 0:
        pde_errors = pde_errors / np.max(pde_errors)
        error_accum += 1.5 * pde_errors
    
    # Sample new points based on error distribution
    probs = np.nan_to_num(error_accum)
    if np.sum(probs) == 0:
        probs = np.ones(N_sample)
    probs = probs / np.sum(probs)
    
    indices = np.random.choice(N_sample, size=N_new, replace=True, p=probs)
    
    # Add perturbation to selected points
    perturb_scale = 0.05
    x_new = x_sample[indices] + np.random.normal(0, perturb_scale * (x_max - x_min), (N_new, 1))
    z_new = z_sample[indices] + np.random.normal(0, perturb_scale * (z_max - z_min), (N_new, 1))
    t_new = t_sample[indices] + np.random.normal(0, perturb_scale * 10, (N_new, 1))
    
    # Clip to domain bounds
    x_new = np.clip(x_new, x_min, x_max)
    z_new = np.clip(z_new, z_min, z_max)
    t_new = np.clip(t_new, 0, time_max)
    
    new_points = torch.tensor(np.hstack((x_new, z_new, t_new)),
                             dtype=torch.float32, device=device, requires_grad=True)
    
    logging.info(f"Adaptive resampling generated {N_new} new points")
    
    return new_points

###############################
# Training Functions
###############################

def train_cycle(model: nn.Module, colloc_points: torch.Tensor,
               top_points: torch.Tensor, left_points: torch.Tensor, right_points: torch.Tensor,
               ic_points: torch.Tensor, ic_saturations: torch.Tensor,
               ert_points: Dict[float, torch.Tensor], ert_saturations: Dict[float, torch.Tensor],
               meas_points: torch.Tensor, target_vals: torch.Tensor,
               hydrus_uptake_handler: HydrusUptakeDataHandler,
               cycle: int, config: Config, monitor: TrainingMonitor,
               checkpoint_manager: CheckpointManager) -> Tuple[float, torch.Tensor]:
    """Train the model for one cycle with comprehensive transpiration loss"""
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                                          patience=200, factor=0.5)
    
    best_loss = float("inf")
    patience_counter = 0
    num_epochs = config.training.num_epochs
    min_epochs = config.training.min_epochs_per_cycle
    
    cycle_start = time.time()
    
    # Get ALL available transpiration times up to 28 days
    available_transpiration_times = []
    pinn_to_hydrus_map = config.get_pinn_to_hydrus_map()
    for pinn_t, hydrus_t in pinn_to_hydrus_map.items():
        if pinn_t <= 28.0 and hydrus_t in hydrus_uptake_handler.uptake_data:  # Only up to 28 days
            available_transpiration_times.append(pinn_t)
    
    available_transpiration_times = sorted(available_transpiration_times)
    logging.info(f"Computing transpiration loss for ALL timesteps up to 28 days: {available_transpiration_times}")
    logging.info(f"Total transpiration timesteps: {len(available_transpiration_times)}")
    
    for epoch in tqdm(range(num_epochs), desc=f"Cycle {cycle+1}", unit="epoch", leave=False):
        epoch_start = time.time()
        optimizer.zero_grad()
        
        # Adjust loss weights based on training progress
        progress = epoch / num_epochs
        if progress < 0.25:
            lambda_pde = 0.5 * 1000.0
            lambda_ic = config.training.ert_weight_ic
            lambda_ert = config.training.ert_weight_data * 0.5
            lambda_meas = 80.0
            lambda_bc = 0.5
            lambda_transp = config.training.lambda_transpiration * 0.5
        elif progress < 0.5:
            lambda_pde = 1.0 * 1000.0
            lambda_ic = config.training.ert_weight_ic * 0.75
            lambda_ert = config.training.ert_weight_data
            lambda_meas = 100.0
            lambda_bc = 1.0
            lambda_transp = config.training.lambda_transpiration
        else:
            lambda_pde = 2.0 * 1000.0
            lambda_ic = config.training.ert_weight_ic * 0.5
            lambda_ert = config.training.ert_weight_data * 1.5
            lambda_meas = 120.0
            lambda_bc = 2.0
            lambda_transp = config.training.lambda_transpiration * 1.5
        
        # Compute losses
        # PDE residual loss
        pde_loss = torch.mean(pde_residual(model, colloc_points, config) ** 2)
        
        # Boundary condition losses
        loss_top = calc_boundary_loss(model, top_points, "top", config)
        loss_left = calc_boundary_loss(model, left_points, "left", config)
        loss_right = calc_boundary_loss(model, right_points, "right", config)
        bc_loss = lambda_bc * (loss_top + loss_left + loss_right)
        
        # Initial condition loss using ERT data
        loss_ic = ert_initial_condition_loss(model, ic_points, ic_saturations)
        
        # ERT data loss for all timesteps
        ert_loss_sum = torch.tensor(0.0, device=device)
        ert_count = 0
        for t, points in ert_points.items():
            if t > 0:  # Skip initial time
                ert_loss_sum += ert_data_loss(model, points, ert_saturations[t])
                ert_count += 1
        
        ert_loss = ert_loss_sum / max(ert_count, 1)
        
        # Measurement loss
        loss_meas = measurement_loss(model, meas_points, target_vals)
        
        # COMPREHENSIVE TRANSPIRATION LOSS - ALL TIMESTEPS EVERY EPOCH
        transpiration_loss_sum = torch.tensor(0.0, device=device)
        transpiration_count = 0
        transpiration_details = []
        
        if available_transpiration_times:
            for t_sample in available_transpiration_times:
                try:
                    t_loss, pinn_total, hydrus_total = transpiration_loss(
                        model, hydrus_uptake_handler, float(t_sample), config
                    )
                    transpiration_loss_sum += t_loss
                    transpiration_count += 1
                    
                    # Store details for logging
                    transpiration_details.append({
                        'time': t_sample,
                        'pinn_total': pinn_total.item(),
                        'hydrus_total': hydrus_total.item(),
                        'loss': t_loss.item()
                    })
                    
                except Exception as e:
                    logging.warning(f"Error computing transpiration loss for t={t_sample}: {e}")
        
        transp_loss = transpiration_loss_sum / max(transpiration_count, 1)
        
        # Total loss INCLUDING comprehensive transpiration
        total_loss = (lambda_pde * pde_loss + 
                     bc_loss + 
                     lambda_ic * loss_ic + 
                     lambda_ert * ert_loss +
                     lambda_meas * loss_meas +
                     lambda_transp * transp_loss)
        
        # Backward pass
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        scheduler.step(total_loss)
        
        # Update metrics
        epoch_time = time.time() - epoch_start
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Enhanced logging with transpiration details
        global_step = cycle * num_epochs + epoch
        
        if epoch % 10 == 0:  # Log every 10 epochs
            metrics = {
                'loss/total': total_loss.item(),
                'loss/pde': pde_loss.item(),
                'loss/ic': loss_ic.item(),
                'loss/ert': ert_loss.item(),
                'loss/measurement': loss_meas.item(),
                'loss/boundary': bc_loss.item(),
                'loss/transpiration': transp_loss.item(),
                'training/grad_norm': grad_norm,
                'training/learning_rate': optimizer.param_groups[0]['lr'],
                'training/epoch_time': epoch_time,
                'transpiration/num_timesteps': transpiration_count
            }
            
            # Add individual transpiration metrics for key timesteps
            if transpiration_details:
                for detail in transpiration_details[::max(1, len(transpiration_details)//5)]:  # Log every 5th timestep
                    t = detail['time']
                    metrics[f'transpiration/pinn_total_t{t:04.1f}'] = detail['pinn_total']
                    metrics[f'transpiration/hydrus_total_t{t:04.1f}'] = detail['hydrus_total']
                    metrics[f'transpiration/loss_t{t:04.1f}'] = detail['loss']
            
            monitor.log_metrics(metrics, global_step)
            
            if epoch % 50 == 0:
                monitor.log_model_gradients(model, global_step)
        
        if epoch % 100 == 0:
            logging.info(f"Cycle {cycle+1} Epoch {epoch}: Total Loss={total_loss.item():.4e}, "
                       f"PDE={pde_loss.item():.4e}, IC={loss_ic.item():.4e}, "
                       f"ERT={ert_loss.item():.4e}, Meas={loss_meas.item():.4e}, "
                       f"BC={bc_loss.item():.4e}, Transp={transp_loss.item():.4e} "
                       f"({transpiration_count} timesteps), Grad={grad_norm:.4e}")
            
            # Log detailed transpiration info every 100 epochs
            if transpiration_details:
                logging.info("Transpiration details:")
                for detail in transpiration_details:
                    logging.info(f"  t={detail['time']:04.1f}: PINN={detail['pinn_total']:.6f}, "
                               f"Hydrus={detail['hydrus_total']:.6f}, Loss={detail['loss']:.6f}")
        
        # Checkpointing
        if epoch % config.training.checkpoint_interval == 0:
            checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, cycle, total_loss.item(), 
                best_loss, colloc_points, config
            )
        
        # Early stopping
        if epoch >= min_epochs and patience_counter > config.training.early_stop_patience:
            logging.info(f"Cycle {cycle+1}: Early stopping at epoch {epoch}")
            break
    
    cycle_duration = time.time() - cycle_start
    logging.info(f"Cycle {cycle+1} complete in {cycle_duration:.2f}s; best loss: {best_loss:.4e}")
    
    return best_loss, colloc_points

###############################
# Data Loading Functions
###############################

def load_hydrus_data(config: Config) -> Tuple[Dict, HydrusDataHandler]:
    """Load Hydrus mesh and pressure data"""
    hydrus_handler = HydrusDataHandler(config)
    
    try:
        # Load mesh
        with open(config.mesh_file, "r") as f:
            mesh_content = f.read()
        hydrus_handler.load_mesh(mesh_content)
        
        # Load pressure
        with open(config.pressure_file, "r") as f:
            pressure_content = f.read()
        hydrus_handler.load_pressure(pressure_content)
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading Hydrus data: {e}")
        raise
    
    # Get initial condition data
    target_time = config.monitoring_time_steps[0]
    domain = hydrus_handler.domain_size
    
    logging.info(f"Domain: x=[{domain['x_min']}, {domain['x_max']}], "
               f"z=[{domain['z_min']}, {domain['z_max']}]")
    
    X, Z, grid_pressures = hydrus_handler.interpolate_data(target_time)
    grid_S = h_to_S(grid_pressures, config)
    
    data = {
        "X": X,
        "Z": Z,
        "x_points": X.flatten(),
        "z_points": Z.flatten(),
        "pressures": grid_pressures.flatten(),
        "saturation": grid_S.flatten(),
        "time": target_time,
        "domain": domain
    }
    
    logging.info(f"Hydrus data: grid {X.shape[1]}x{X.shape[0]}, "
               f"pressure range [{np.nanmin(grid_pressures):.3f}, {np.nanmax(grid_pressures):.3f}], "
               f"saturation range [{np.nanmin(grid_S):.3f}, {np.nanmax(grid_S):.3f}]")
    
    return data, hydrus_handler

###############################
# Main Training Function
###############################

def train_pinn_with_adaptive_sampling(config: Config) -> Tuple[nn.Module, Dict, Any]:
    """Main training function with adaptive sampling, ERT data, and transpiration loss"""
    logging.info("Starting PINN training with adaptive sampling, ERT data, and transpiration loss...")
    logging.info(f"Domain: z ∈ [{config.domain.z_min}, {config.domain.z_max}] m (reduced domain)")
    
    # Create output directories
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    # Initialize monitoring and checkpointing
    monitor = TrainingMonitor(config)
    checkpoint_manager = CheckpointManager(config.training.checkpoint_dir)
    
    # Save configuration
    config.save_json(os.path.join(config.training.output_dir, "config.json"))
    
    # Load Hydrus data
    hydrus_data, hydrus_handler = load_hydrus_data(config)
    hydrus_interp = HydrusInterpolator(hydrus_data, config)
    
    # Get monitoring data
    pressure_values = hydrus_handler.get_values_at_locations(
        config.monitoring_locations, config.monitoring_time_steps
    )
    meas_points, target_vals = generate_measurement_points(
        config.monitoring_locations, config.monitoring_time_steps, 
        pressure_values, config
    )
    
    # Load ERT data
    ert_handler = ERTDataHandler(config)
    ert_handler.load_ert_data()
    ert_handler.convert_to_saturation()
    
    # Prepare ERT data for training
    ert_points_dict, ert_saturations_dict = prepare_ert_data_for_training(
        ert_handler.ert_data, config
    )
    
    # Load Hydrus uptake data for transpiration loss
    hydrus_uptake_handler = HydrusUptakeDataHandler(config)
    try:
        hydrus_uptake_handler.load_uptake_data()
        logging.info("Hydrus uptake data loaded successfully for transpiration loss")
        logging.info(f"Available uptake timesteps: {sorted(hydrus_uptake_handler.uptake_data.keys())}")
    except Exception as e:
        logging.warning(f"Could not load Hydrus uptake data: {e}")
        logging.warning("Transpiration loss will be disabled")
        config.training.lambda_transpiration = 0.0
    
    # Get initial condition from ERT
    if 0 not in ert_points_dict:
        raise ValueError("Initial time (t=0) not found in ERT data")
    
    ic_points = ert_points_dict[0]
    ic_saturations = ert_saturations_dict[0]
    logging.info(f"Using {len(ic_points)} ERT points at t=0 as initial condition")
    
    # Create model
    model = PINNWithUptake(config).to(device)
    
    # Check for existing checkpoint
    start_cycle = 0
    best_loss = float("inf")
    colloc_points = None
    
    checkpoint = checkpoint_manager.load_latest_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_cycle = checkpoint['cycle'] + 1
        best_loss = checkpoint['best_loss']
        colloc_points = checkpoint['colloc_points'].to(device)
        colloc_points.requires_grad_(True)
        logging.info(f"Resumed from cycle {start_cycle} with best loss {best_loss:.4e}")
    else:
        # Generate initial collocation points
        colloc_points, top_points, left_points, right_points = generate_collocation_points(config)
    
    # Training loop
    loss_history = {"total": []}
    global_best_loss = best_loss
    best_model_state = None
    cycles_without_improvement = 0
    total_start = time.time()
    
    try:
        for cycle in range(start_cycle, config.training.num_cycles):
            logging.info(f"\n{'='*60}")
            logging.info(f"Training Cycle {cycle+1}/{config.training.num_cycles}")
            logging.info(f"{'='*60}")
            
            # Generate boundary points (they change each cycle)
            if cycle > start_cycle or colloc_points is None:
                _, top_points, left_points, right_points = generate_collocation_points(config)
            
            # Train for one cycle
            cycle_best_loss, colloc_points = train_cycle(
                model, colloc_points, top_points, left_points, right_points,
                ic_points, ic_saturations, ert_points_dict, ert_saturations_dict,
                meas_points, target_vals, hydrus_uptake_handler, cycle, config, 
                monitor, checkpoint_manager
            )
            
            loss_history["total"].append(cycle_best_loss)
            
            # Update global best
            if cycle_best_loss < global_best_loss:
                global_best_loss = cycle_best_loss
                best_model_state = model.state_dict()
                cycles_without_improvement = 0
                logging.info(f"New global best loss: {global_best_loss:.4e}")
            else:
                cycles_without_improvement += 1
                logging.info(f"No improvement; count={cycles_without_improvement}")
            
            # Adaptive resampling (except after last cycle)
            if cycle < config.training.num_cycles - 1:
                new_points = compute_residuals_and_resample(
                    model, hydrus_handler, ert_handler, config,
                    N_sample=config.training.resampling_sample,
                    N_new=config.training.resampling_new
                )
                colloc_points = torch.cat([colloc_points, new_points], dim=0)
                logging.info(f"Updated collocation points: {colloc_points.shape[0]}")
            
            # Clear GPU cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Early termination
            if cycles_without_improvement >= config.training.max_cycles_without_improvement:
                logging.info("No improvement in multiple cycles. Terminating training.")
                break
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    finally:
        # Finalize
        total_duration = time.time() - total_start
        logging.info(f"\nTraining complete in {total_duration:.2f}s")
        logging.info(f"Global best loss: {global_best_loss:.4e}")
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Save final model
        final_model_path = os.path.join(config.training.output_dir, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to: {final_model_path}")
        
        # Save loss history
        loss_history_path = os.path.join(config.training.output_dir, "loss_history.pkl")
        with open(loss_history_path, 'wb') as f:
            pickle.dump(loss_history, f)
        
        # Close monitor
        monitor.close()
    
    return model, loss_history, {
        'hydrus_data': hydrus_data,
        'hydrus_handler': hydrus_handler,
        'hydrus_interp': hydrus_interp,
        'ert_handler': ert_handler,
        'hydrus_uptake_handler': hydrus_uptake_handler,
        'pressure_values': pressure_values
    }

###############################
# Visualization Functions
###############################

class PlotManager:
    """Manage plot creation with consistent styling"""
    def __init__(self, config: Config, style: str = 'seaborn-v0_8'):
        self.config = config
        if style in plt.style.available:
            plt.style.use(style)
        self.setup_plot_params()
    
    def setup_plot_params(self):
        """Set up matplotlib parameters"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.grid': True,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def save_or_show(self, fig: plt.Figure, fname: Optional[str] = None) -> None:
        """Save figure or show interactively"""
        if fname:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            fig.savefig(fname, bbox_inches="tight")
            logging.info(f"Figure saved to {fname}")
            plt.close(fig)
        else:
            plt.show()
    
    def plot_contours(self, ax: plt.Axes, X: np.ndarray, Z: np.ndarray, 
                     data: np.ndarray, title: str, cmap: str = "viridis",
                     clabel: str = "", levels: Optional[np.ndarray] = None) -> Any:
        """Create contour plot with consistent styling"""
        if levels is None:
            levels = 20
        
        cf = ax.contourf(X, Z, data, levels=levels, cmap=cmap, extend='both')
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(self.config.domain.z_min, self.config.domain.z_max)
        
        cbar = plt.colorbar(cf, ax=ax, label=clabel)
        return cf

def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute RMSE between predictions and true values"""
    mask = ~(np.isnan(pred) | np.isnan(true))
    if np.sum(mask) == 0:
        return np.nan
    return np.sqrt(np.mean((pred[mask] - true[mask])**2))

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive comparison metrics"""
    mask = ~(np.isnan(pred) | np.isnan(true))
    pred_clean = pred[mask]
    true_clean = true[mask]
    
    if len(pred_clean) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "n_points": 0}
    
    rmse = np.sqrt(np.mean((pred_clean - true_clean)**2))
    mae = np.mean(np.abs(pred_clean - true_clean))
    
    ss_res = np.sum((true_clean - pred_clean)**2)
    ss_tot = np.sum((true_clean - np.mean(true_clean))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    bias = np.mean(pred_clean - true_clean)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias,
        "n_points": len(pred_clean)
    }

def plot_loss_history(loss_history: Dict, plot_manager: PlotManager, 
                     save_fig: Optional[str] = None) -> None:
    """Plot training loss history"""
    epochs = np.arange(len(loss_history["total"]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss_history["total"], "ko-", linewidth=2, markersize=6)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training Loss History (Reduced Domain)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plot_manager.save_or_show(fig, save_fig)

def plot_solution_with_reference(model: nn.Module, X: np.ndarray, Z: np.ndarray, t_val: float,
                                ref_data: Dict, ref_interp: HydrusInterpolator,
                                ert_handler: Optional[ERTDataHandler],
                                monitoring_locations: List[Tuple[float, float]],
                                plot_manager: PlotManager,
                                save_fig: Optional[str] = None) -> None:
    """Plot comparison at given PINN time with reference data"""
    config = plot_manager.config
    
    t_array = t_val * np.ones_like(X)
    input_slice = np.stack([X, Z, t_array], axis=-1).reshape(-1, 3)
    input_tensor = torch.tensor(input_slice, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        S_pred, _ = model(input_tensor)
    S_pred = S_pred.cpu().numpy().reshape(X.shape)
    
    ref_time = t_val + config.domain.pinn_time_offset
    x_grid = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
    z_grid = torch.tensor(Z.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
    S_ref = ref_interp.interpolate(x_grid, z_grid).cpu().numpy().reshape(X.shape)
    
    S_ert = None
    if ert_handler is not None and ref_time in ert_handler.ert_data:
        ert_coords = ert_handler.ert_data[ref_time]['coords']
        ert_sat = ert_handler.ert_data[ref_time]['saturation']
        
        points = ert_coords
        values = ert_sat
        xi = np.column_stack([X.flatten(), Z.flatten()])
        S_ert = griddata(points, values, xi, method='linear').reshape(X.shape)
    
    rmse_hydrus = compute_rmse(S_pred, S_ref)
    rmse_ert = compute_rmse(S_pred, S_ert) if S_ert is not None else None
    
    if S_ert is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    levels_sat = np.linspace(0, 1, 21)
    
    plt.sca(axes[0])
    plot_manager.plot_contours(axes[0], X, Z, S_pred, 
                              f"PINN Prediction (t={t_val:.1f} d)", 
                              "viridis", "Saturation", levels_sat)
    
    plt.sca(axes[1])
    plot_manager.plot_contours(axes[1], X, Z, S_ref, 
                              f"Hydrus Data (t={ref_time:.1f} d)\nRMSE={rmse_hydrus:.2e}", 
                              "viridis", "Saturation", levels_sat)
    
    if S_ert is not None:
        plt.sca(axes[2])
        plot_manager.plot_contours(axes[2], X, Z, S_ert, 
                                  f"ERT Saturation (t={ref_time:.1f} d)\nRMSE={rmse_ert:.2e}", 
                                  "viridis", "Saturation", levels_sat)
    
    if monitoring_locations:
        filtered_locations = [(x, z) for x, z in monitoring_locations 
                             if config.domain.z_min <= z <= config.domain.z_max]
        names = ["Shallow", "Medium", "Deep"]
        for i, ((x, z), name) in enumerate(zip(filtered_locations, names)):
            for ax in axes:
                ax.plot(x, z, "ro", markersize=8)
                ax.text(x + 0.2, z, name, fontweight="bold", color="red")
    
    plt.tight_layout()
    plot_manager.save_or_show(fig, save_fig)

def plot_comparison_at_time(model: nn.Module, hydrus_handler: HydrusDataHandler,
                           ert_handler: ERTDataHandler, domain: Dict[str, float],
                           t: float, plot_manager: PlotManager,
                           save_fig: Optional[str] = None) -> None:
    """Create detailed comparison for specified PINN time"""
    config = plot_manager.config
    
    # Use consistent grid size for both PINN and Hydrus
    num_x, num_z = 50, 30  # Define grid size once
    
    x_vals = np.linspace(domain["x_min"], domain["x_max"], num_x)
    z_vals = np.linspace(config.domain.z_min, config.domain.z_max, num_z)
    X, Z = np.meshgrid(x_vals, z_vals)
    
    t_array = t * np.ones_like(X)
    input_slice = np.stack([X, Z, t_array], axis=-1).reshape(-1, 3)
    input_tensor = torch.tensor(input_slice, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        S_pred, _ = model(input_tensor)
    S_pred = S_pred.cpu().numpy().reshape(X.shape)
    
    hydrus_time = t + config.domain.pinn_time_offset
    # Pass the same grid size to hydrus interpolation
    X_h, Z_h, pressure_grid = hydrus_handler.interpolate_data(
        hydrus_time, 
        num_x=num_x,  # Use same grid size
        num_z=num_z   # Use same grid size
    )
    S_hydrus = h_to_S(pressure_grid, config)
    
    S_ert = None
    if hydrus_time in ert_handler.ert_data:
        ert_coords = ert_handler.ert_data[hydrus_time]['coords']
        ert_sat = ert_handler.ert_data[hydrus_time]['saturation']
        
        points = ert_coords
        values = ert_sat
        xi = np.column_stack([X.flatten(), Z.flatten()])
        S_ert = griddata(points, values, xi, method='linear').reshape(X.shape)
    
    diff_hydrus = S_pred - S_hydrus
    rmse_hydrus = compute_rmse(S_pred, S_hydrus)
    
    if S_ert is not None:
        diff_ert = S_pred - S_ert
        rmse_ert = compute_rmse(S_pred, S_ert)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        levels_sat = np.linspace(0, 1, 21)
        levels_diff = np.linspace(-0.2, 0.2, 21)
        
        plot_manager.plot_contours(axes[0], X, Z, S_pred, 
                                  f"PINN Prediction (t={t:.1f} d)", 
                                  "viridis", "Saturation", levels_sat)
        
        plot_manager.plot_contours(axes[1], X_h, Z_h, S_hydrus, 
                                  f"Hydrus Data (t={hydrus_time:.1f} d)\nRMSE={rmse_hydrus:.2e}", 
                                  "viridis", "Saturation", levels_sat)
        
        plot_manager.plot_contours(axes[2], X, Z, S_ert, 
                                  f"ERT Saturation (t={hydrus_time:.1f} d)\nRMSE={rmse_ert:.2e}", 
                                  "viridis", "Saturation", levels_sat)
        
        plot_manager.plot_contours(axes[3], X, Z, diff_ert, 
                                  f"PINN - ERT Difference", 
                                  "RdBu_r", "Diff", levels_diff)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        levels_sat = np.linspace(0, 1, 21)
        levels_diff = np.linspace(-0.2, 0.2, 21)
        
        plot_manager.plot_contours(axes[0], X, Z, S_pred, 
                                  f"PINN Prediction (t={t:.1f} d)", 
                                  "viridis", "Saturation", levels_sat)
        
        plot_manager.plot_contours(axes[1], X_h, Z_h, S_hydrus, 
                                  f"Hydrus Data (t={hydrus_time:.1f} d)", 
                                  "viridis", "Saturation", levels_sat)
        
        plot_manager.plot_contours(axes[2], X, Z, diff_hydrus, 
                                  f"Difference (RMSE={rmse_hydrus:.2e})", 
                                  "RdBu_r", "Diff", levels_diff)
    
    plt.tight_layout()
    plot_manager.save_or_show(fig, save_fig)

def plot_uptake(model: nn.Module, domain: Dict[str, float], t: float,
               config: Config, plot_manager: PlotManager,
               save_fig: Optional[str] = None) -> None:
    """Contour plot of learned uptake with physical aspect ratio"""
    x_range = domain["x_max"] - domain["x_min"]
    z_range = config.domain.z_max - config.domain.z_min
    aspect_ratio = x_range / z_range
    
    fig_height = 4
    fig_width = fig_height * aspect_ratio
    if fig_width > 16:
        fig_width = 16
        fig_height = fig_width / aspect_ratio
    
    x_vals = np.linspace(domain["x_min"], domain["x_max"], 50)
    z_vals = np.linspace(config.domain.z_min, config.domain.z_max, 30)
    X, Z = np.meshgrid(x_vals, z_vals)
    
    t_arr = t * np.ones_like(X)
    pts = np.stack([X, Z, t_arr], axis=-1).reshape(-1, 3)
    pts_tensor = torch.tensor(pts, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        _, U_pred = model(pts_tensor)
    U = U_pred.cpu().numpy().reshape(X.shape)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    levels = np.linspace(0.0, config.soil.RU_max, 21)
    
    u_min, u_max = np.min(U), np.max(U)
    logging.info(f"Uptake at t={t:.1f}: range [{u_min:.6f}, {u_max:.6f}]")
    
    cf = ax.contourf(X, Z, U, levels=levels, cmap="plasma", extend='max')
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Learned Root Uptake U(x,z) at t={t:.1f} d\n" + 
                f"(Domain: {x_range:.1f}m × {z_range:.1f}m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_xlim(domain["x_min"], domain["x_max"])
    ax.set_ylim(config.domain.z_min, config.domain.z_max)
    
    cbar = fig.colorbar(cf, ax=ax, label="Uptake (m/day)", shrink=0.8)
    cbar_ticks = np.linspace(0, config.soil.RU_max, 6)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in cbar_ticks])
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plot_manager.save_or_show(fig, save_fig)

def plot_cross_sections(model: nn.Module, hydrus_handler: HydrusDataHandler,
                       ert_handler: ERTDataHandler, domain: Dict[str, float],
                       t: float, config: Config, plot_manager: PlotManager,
                       save_fig: Optional[str] = None) -> None:
    """Create cross-section plots at different x and z locations"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    hydrus_time = t + config.domain.pinn_time_offset
    
    x_positions = [2.0, 4.5, 7.0]
    colors = ['blue', 'green', 'red']
    
    # Vertical profiles
    z_vals = np.linspace(config.domain.z_min, config.domain.z_max, 100)
    for x_pos, color in zip(x_positions, colors):
        points = np.column_stack([np.full(100, x_pos), z_vals, np.full(100, t)])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            S_pinn, _ = model(points_tensor)
        S_pinn = S_pinn.cpu().numpy().flatten()
        
        ax1.plot(S_pinn, z_vals, '-', color=color, linewidth=2, label=f'PINN x={x_pos}m')
        
        # Get Hydrus data for comparison
        try:
            # Create a narrow vertical slice for interpolation
            x_slice = np.linspace(x_pos - 0.05, x_pos + 0.05, 5)
            z_slice = z_vals
            X_slice, Z_slice = np.meshgrid(x_slice, z_slice)
            
            # Get interpolated pressure at this slice
            pressure_at_slice = []
            for z_val in z_vals:
                try:
                    # Get pressure at specific (x, z) point
                    loc_pressure = hydrus_handler.get_values_at_locations(
                        [(x_pos, z_val)], [hydrus_time]
                    )
                    if hydrus_time in loc_pressure and len(loc_pressure[hydrus_time]) > 0:
                        pressure_at_slice.append(loc_pressure[hydrus_time][0])
                    else:
                        pressure_at_slice.append(np.nan)
                except:
                    pressure_at_slice.append(np.nan)
            
            pressure_at_slice = np.array(pressure_at_slice)
            S_hydrus = h_to_S(pressure_at_slice, config)
            
            # Only plot if we have valid data
            valid_mask = ~np.isnan(S_hydrus)
            if np.any(valid_mask):
                ax1.plot(S_hydrus[valid_mask], z_vals[valid_mask], '--', color=color, 
                        linewidth=1.5, label=f'Hydrus x={x_pos}m')
        except Exception as e:
            logging.debug(f"Could not get Hydrus data for vertical profile at x={x_pos}: {e}")
    
    ax1.set_xlabel('Saturation')
    ax1.set_ylabel('Depth z (m)')
    ax1.set_title(f'Vertical Profiles at t={t:.1f} days')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Horizontal profiles
    z_positions = [4.6, 5.0, 5.4]
    x_vals = np.linspace(domain["x_min"], domain["x_max"], 100)
    
    for z_pos, color in zip(z_positions, colors):
        points = np.column_stack([x_vals, np.full(100, z_pos), np.full(100, t)])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            S_pinn, _ = model(points_tensor)
        S_pinn = S_pinn.cpu().numpy().flatten()
        
        ax2.plot(x_vals, S_pinn, '-', color=color, linewidth=2, label=f'PINN z={z_pos}m')
        
        try:
            X_h, Z_h, pressure_grid = hydrus_handler.interpolate_data(
                hydrus_time, num_x=100, num_z=2
            )
            S_hydrus = h_to_S(pressure_grid[0, :], config)
            ax2.plot(x_vals, S_hydrus, '--', color=color, linewidth=1.5, label=f'Hydrus z={z_pos}m')
        except:
            pass
    
    ax2.set_xlabel('Horizontal distance x (m)')
    ax2.set_ylabel('Saturation')
    ax2.set_title(f'Horizontal Profiles at t={t:.1f} days')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Uptake profiles
    for x_pos, color in zip(x_positions, colors):
        points = np.column_stack([np.full(100, x_pos), z_vals, np.full(100, t)])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            _, U_pinn = model(points_tensor)
        U_pinn = U_pinn.cpu().numpy().flatten()
        
        ax3.plot(U_pinn * 1000, z_vals, '-', color=color, linewidth=2, label=f'x={x_pos}m')
    
    ax3.set_xlabel('Uptake (mm/day)')
    ax3.set_ylabel('Depth z (m)')
    ax3.set_title(f'Uptake Profiles at t={t:.1f} days')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Error profiles
    for x_pos, color in zip(x_positions, colors):
        points = np.column_stack([np.full(100, x_pos), z_vals, np.full(100, t)])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            S_pinn, _ = model(points_tensor)
        S_pinn = S_pinn.cpu().numpy().flatten()
        
        try:
            X_h, Z_h, pressure_grid = hydrus_handler.interpolate_data(
                hydrus_time, num_x=2, num_z=100
            )
            S_hydrus = h_to_S(pressure_grid[:, 0], config)
            error = S_pinn - S_hydrus
            ax4.plot(error, z_vals, '-', color=color, linewidth=2, label=f'x={x_pos}m')
        except:
            pass
    
    ax4.set_xlabel('Error (PINN - Hydrus)')
    ax4.set_ylabel('Depth z (m)')
    ax4.set_title(f'Error Profiles at t={t:.1f} days')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Cross-Section Analysis at PINN t={t:.1f} days', fontsize=14)
    plt.tight_layout()
    plot_manager.save_or_show(fig, save_fig)

def plot_scatter_comparison(model: nn.Module, hydrus_handler: HydrusDataHandler,
                           domain: Dict[str, float], t: float, config: Config,
                           plot_manager: PlotManager, n_points: int = 2000,
                           save_fig: Optional[str] = None) -> Dict[str, float]:
    """Create scatter plot comparing Hydrus vs PINN predictions"""
    hydrus_time = t + config.domain.pinn_time_offset
    
    np.random.seed(42)
    x_pts = np.random.uniform(domain["x_min"], domain["x_max"], n_points)
    z_pts = np.random.uniform(config.domain.z_min, config.domain.z_max, n_points)
    
    try:
        X_h, Z_h, pressure_grid = hydrus_handler.interpolate_data(
            hydrus_time, num_x=50, num_z=30
        )
        S_hydrus_grid = h_to_S(pressure_grid, config)
        
        hydrus_points = np.column_stack([X_h.flatten(), Z_h.flatten()])
        hydrus_values = S_hydrus_grid.flatten()
        
        valid_mask = ~np.isnan(hydrus_values)
        hydrus_points_clean = hydrus_points[valid_mask]
        hydrus_values_clean = hydrus_values[valid_mask]
        
        sample_points = np.column_stack([x_pts, z_pts])
        S_hydrus_interp = griddata(hydrus_points_clean, hydrus_values_clean,
                                  sample_points, method='linear')
    except Exception as e:
        logging.error(f"Error getting Hydrus data for scatter plot at t={t}: {e}")
        return {}
    
    t_array = np.full(n_points, t)
    pinn_input = np.column_stack([x_pts, z_pts, t_array])
    pinn_tensor = torch.tensor(pinn_input, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        S_pinn, _ = model(pinn_tensor)
    S_pinn_vals = S_pinn.cpu().numpy().flatten()
    
    metrics = compute_metrics(S_pinn_vals, S_hydrus_interp)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    mask = ~(np.isnan(S_pinn_vals) | np.isnan(S_hydrus_interp))
    S_pinn_clean = S_pinn_vals[mask]
    S_hydrus_clean = S_hydrus_interp[mask]
    
    if len(S_pinn_clean) > 0:
        scatter = ax.scatter(S_hydrus_clean, S_pinn_clean, alpha=0.6, s=20,
                           c=np.sqrt((x_pts[mask] - domain["x_min"])**2 +
                                   (z_pts[mask] - config.domain.z_min)**2),
                           cmap='viridis', rasterized=True)
        
        min_val = min(np.min(S_hydrus_clean), np.min(S_pinn_clean))
        max_val = max(np.max(S_hydrus_clean), np.max(S_pinn_clean))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
               label='Perfect Agreement')
        
        ax.set_xlim(min_val - 0.05, max_val + 0.05)
        ax.set_ylim(min_val - 0.05, max_val + 0.05)
        ax.set_xlabel('Hydrus Saturation')
        ax.set_ylabel('PINN Saturation')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Distance from origin (m)')
        
        metrics_text = (f'RMSE = {metrics["rmse"]:.4f}\n'
                       f'MAE = {metrics["mae"]:.4f}\n'
                       f'R² = {metrics["r2"]:.4f}\n'
                       f'Bias = {metrics["bias"]:.4f}\n'
                       f'N = {metrics["n_points"]}')
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.8))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax.set_title(f'PINN vs Hydrus Saturation Comparison\nt={t:.1f} days', fontsize=14)
    
    plt.tight_layout()
    plot_manager.save_or_show(fig, save_fig)
    
    return metrics

def plot_scatter_comparison_ert(model: nn.Module, ert_handler: ERTDataHandler,
                               domain: Dict[str, float], t: float, config: Config,
                               plot_manager: PlotManager,
                               save_fig: Optional[str] = None) -> Dict[str, float]:
    """Create scatter plot comparing ERT vs PINN predictions"""
    hydrus_time = t + config.domain.pinn_time_offset
    
    if hydrus_time not in ert_handler.ert_data:
        logging.warning(f"ERT data not available for time {hydrus_time}")
        return {}
    
    ert_data = ert_handler.ert_data[hydrus_time]
    ert_coords = ert_data['coords']
    ert_sat = ert_data['saturation']
    
    n_points = len(ert_coords)
    t_array = np.full(n_points, t)
    pinn_input = np.column_stack([ert_coords, t_array])
    pinn_tensor = torch.tensor(pinn_input, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        S_pinn, _ = model(pinn_tensor)
    S_pinn_vals = S_pinn.cpu().numpy().flatten()
    
    metrics = compute_metrics(S_pinn_vals, ert_sat)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    mask = ~(np.isnan(S_pinn_vals) | np.isnan(ert_sat))
    S_pinn_clean = S_pinn_vals[mask]
    S_ert_clean = ert_sat[mask]
    ert_coords_clean = ert_coords[mask]
    
    if len(S_pinn_clean) > 0:
        scatter = ax.scatter(S_ert_clean, S_pinn_clean, alpha=0.6, s=20,
                           c=ert_coords_clean[:, 1], cmap='plasma', rasterized=True)
        
        min_val = min(np.min(S_ert_clean), np.min(S_pinn_clean))
        max_val = max(np.max(S_ert_clean), np.max(S_pinn_clean))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
               label='Perfect Agreement')
        
        ax.set_xlim(min_val - 0.05, max_val + 0.05)
        ax.set_ylim(min_val - 0.05, max_val + 0.05)
        ax.set_xlabel('ERT-derived Saturation')
        ax.set_ylabel('PINN Saturation')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Depth z (m)')
        
        metrics_text = (f'RMSE = {metrics["rmse"]:.4f}\n'
                       f'MAE = {metrics["mae"]:.4f}\n'
                       f'R² = {metrics["r2"]:.4f}\n'
                       f'Bias = {metrics["bias"]:.4f}\n'
                       f'N = {metrics["n_points"]}')
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.8))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax.set_title(f'PINN vs ERT Saturation Comparison\nt={t:.1f} days', fontsize=14)
    
    plt.tight_layout()
    plot_manager.save_or_show(fig, save_fig)
    
    return metrics

def plot_monitoring_timeseries(model: nn.Module, hydrus_handler: HydrusDataHandler,
                              domain: Dict[str, float], pinn_times: List[float],
                              config: Config, plot_manager: PlotManager,
                              save_fig: Optional[str] = None) -> None:
    """Plot time series of saturation at monitoring locations"""
    filtered_locations = [(x, z) for x, z in config.monitoring_locations 
                         if config.domain.z_min <= z <= config.domain.z_max]
    n_locations = len(filtered_locations)
    
    fig, axes = plt.subplots(n_locations, 1, figsize=(12, 4*n_locations), sharex=True)
    if n_locations == 1:
        axes = [axes]
    
    location_names = ["Shallow", "Medium", "Deep"]
    
    for i, ((x, z), name) in enumerate(zip(filtered_locations, location_names)):
        ax = axes[i]
        
        pinn_saturations = []
        for t in pinn_times:
            point = torch.tensor([[x, z, t]], dtype=torch.float32, device=device)
            with torch.no_grad():
                S, _ = model(point)
            pinn_saturations.append(S.cpu().numpy()[0, 0])
        
        hydrus_saturations = []
        hydrus_times = [t + config.domain.pinn_time_offset for t in pinn_times]
        
        for h_time in hydrus_times:
            if h_time in hydrus_handler.time_steps:
                pressure_values = hydrus_handler.get_values_at_locations([(x, z)], [h_time])
                if h_time in pressure_values and len(pressure_values[h_time]) > 0:
                    h_val = pressure_values[h_time][0]
                    S_val = h_to_S(np.array([h_val]), config)[0]
                    hydrus_saturations.append(S_val)
                else:
                    hydrus_saturations.append(np.nan)
            else:
                hydrus_saturations.append(np.nan)
        
        ax.plot(pinn_times, pinn_saturations, 'b-', linewidth=2, marker='o',
                markersize=4, label='PINN')
        ax.plot(pinn_times, hydrus_saturations, 'r--', linewidth=2, marker='s',
                markersize=4, label='Hydrus')
        
        ax.set_ylabel('Saturation')
        ax.set_title(f'{name} Location: x={x:.1f}m, z={z:.1f}m')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
        
        valid_mask = ~np.isnan(hydrus_saturations)
        if np.any(valid_mask):
            rmse = np.sqrt(np.mean((np.array(pinn_saturations)[valid_mask] -
                                  np.array(hydrus_saturations)[valid_mask])**2))
            ax.text(0.02, 0.95, f'RMSE = {rmse:.4f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('PINN Time (days)')
    
    plt.suptitle('Saturation Time Series at Monitoring Locations', fontsize=14)
    plt.tight_layout()
    plot_manager.save_or_show(fig, save_fig)

def create_metrics_summary_plot(hydrus_metrics: Dict[float, Dict[str, float]],
                               ert_metrics: Dict[float, Dict[str, float]],
                               plot_manager: PlotManager,
                               save_fig: Optional[str] = None) -> None:
    """Create summary plots showing metrics evolution over time"""
    if not hydrus_metrics and not ert_metrics:
        logging.warning("No metrics data available for summary plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    hydrus_times = sorted(hydrus_metrics.keys()) if hydrus_metrics else []
    ert_times = sorted(ert_metrics.keys()) if ert_metrics else []
    
    metrics_names = ['rmse', 'mae', 'r2', 'bias']
    metrics_labels = ['RMSE', 'MAE', 'R²', 'Bias']
    
    for i, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
        ax = axes[i]
        
        if hydrus_times:
            hydrus_vals = [hydrus_metrics[t].get(metric, np.nan) for t in hydrus_times]
            valid_mask = ~np.isnan(hydrus_vals)
            if np.any(valid_mask):
                ax.plot(np.array(hydrus_times)[valid_mask], np.array(hydrus_vals)[valid_mask],
                       'o-', label='vs Hydrus', linewidth=2, markersize=6)
        
        if ert_times:
            ert_vals = [ert_metrics[t].get(metric, np.nan) for t in ert_times]
            valid_mask = ~np.isnan(ert_vals)
            if np.any(valid_mask):
                ax.plot(np.array(ert_times)[valid_mask], np.array(ert_vals)[valid_mask],
                       's-', label='vs ERT', linewidth=2, markersize=6)
        
        ax.set_xlabel('PINN Time (days)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if metric == 'r2':
            ax.set_ylim(-1.0, 1.1)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('PINN Performance Metrics Over Time', fontsize=16)
    plt.tight_layout()
    plot_manager.save_or_show(fig, save_fig)

def plot_ert_data(ert_handler: ERTDataHandler, timestep: float,
                 domain: Dict[str, float], config: Config,
                 plot_manager: PlotManager, save_fig: Optional[str] = None) -> None:
    """Visualize ERT data for specific timestep"""
    if timestep not in ert_handler.ert_data:
        logging.warning(f"ERT data for timestep {timestep} not found")
        return
    
    ert_data = ert_handler.ert_data[timestep]
    coords = ert_data['coords']
    resistivity = ert_data['resistivity']
    saturation = ert_data['saturation']
    
    x_vals = np.linspace(domain["x_min"], domain["x_max"], 50)
    z_vals = np.linspace(config.domain.z_min, config.domain.z_max, 30)
    X, Z = np.meshgrid(x_vals, z_vals)
    
    points = coords
    res_grid = griddata(points, resistivity, (X, Z), method='linear')
    sat_grid = griddata(points, saturation, (X, Z), method='linear')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = ax1.contourf(X, Z, res_grid, 20, cmap='viridis')
    ax1.set_title(f"ERT Resistivity at t={timestep} d")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_ylim(config.domain.z_min, config.domain.z_max)
    ax1.set_aspect('equal')
    fig.colorbar(im1, ax=ax1, label='Resistivity (Ω·m)')
    
    im2 = ax2.contourf(X, Z, sat_grid, 20, cmap='viridis')
    ax2.set_title(f"ERT-derived Effective Saturation (Se) at t={timestep} d")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_ylim(config.domain.z_min, config.domain.z_max)
    ax2.set_aspect('equal')
    fig.colorbar(im2, ax=ax2, label='Saturation')
    
    plt.figtext(0.5, 0.01, f'Reduced domain z ∈ [{config.domain.z_min}, {config.domain.z_max}] m',
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plot_manager.save_or_show(fig, save_fig)

def create_comprehensive_plots(model: nn.Module, hydrus_handler: HydrusDataHandler,
                             ert_handler: ERTDataHandler, hydrus_data: Dict,
                             hydrus_interp: HydrusInterpolator, domain: Dict[str, float],
                             config: Config, plot_manager: PlotManager,
                             save_fig_dir: Optional[str] = None) -> None:
    """Create comprehensive plots for ALL timesteps"""
    logging.info("="*80)
    logging.info("CREATING COMPREHENSIVE PLOTS FOR ALL TIMESTEPS")
    logging.info("="*80)
    
    all_pinn_times = sorted(config.get_pinn_to_hydrus_map().keys())
    logging.info(f"Will create plots for {len(all_pinn_times)} timesteps: {all_pinn_times}")
    
    all_hydrus_metrics = {}
    all_ert_metrics = {}
    
    if save_fig_dir:
        subdirs = ['contours', 'comparisons', 'uptake', 'scatter', 'ert_data', 'cross_sections']
        for subdir in subdirs:
            os.makedirs(os.path.join(save_fig_dir, subdir), exist_ok=True)
    
    for idx, t in enumerate(all_pinn_times):
        hydrus_time = t + config.domain.pinn_time_offset
        logging.info(f"\n[{idx+1}/{len(all_pinn_times)}] Processing PINN time t={t} (Hydrus t={hydrus_time})...")
        
        # 1. Saturation contour comparison
        fname_contour = os.path.join(save_fig_dir, 'contours', f'saturation_contours_t{t:03d}.png') if save_fig_dir else None
        try:
            plot_solution_with_reference(
                model, hydrus_data["X"], hydrus_data["Z"], t,
                hydrus_data, hydrus_interp, ert_handler, config.monitoring_locations,
                plot_manager, fname_contour
            )
        except Exception as e:
            logging.error(f"Error creating contour plot at t={t}: {e}")
        
        # 2. Detailed comparison with difference maps
        fname_comp = os.path.join(save_fig_dir, 'comparisons', f'detailed_comparison_t{t:03d}.png') if save_fig_dir else None
        try:
            plot_comparison_at_time(model, hydrus_handler, ert_handler, domain, t, 
                                  plot_manager, fname_comp)
        except Exception as e:
            logging.error(f"Error creating comparison plot at t={t}: {e}")
        
        # 3. Uptake field visualization
        fname_uptake = os.path.join(save_fig_dir, 'uptake', f'uptake_field_t{t:03d}.png') if save_fig_dir else None
        try:
            plot_uptake(model, domain, t, config, plot_manager, fname_uptake)
        except Exception as e:
            logging.error(f"Error creating uptake plot at t={t}: {e}")
        
        # 4. Scatter plot: PINN vs Hydrus
        fname_scatter_h = os.path.join(save_fig_dir, 'scatter', f'scatter_hydrus_t{t:03d}.png') if save_fig_dir else None
        try:
            metrics_h = plot_scatter_comparison(model, hydrus_handler, domain, t, config,
                                              plot_manager, n_points=3000, save_fig=fname_scatter_h)
            if metrics_h:
                all_hydrus_metrics[t] = metrics_h
        except Exception as e:
            logging.error(f"Error creating Hydrus scatter plot at t={t}: {e}")
        
        # 5. Scatter plot: PINN vs ERT
        if hydrus_time in ert_handler.ert_data:
            fname_scatter_e = os.path.join(save_fig_dir, 'scatter', f'scatter_ert_t{t:03d}.png') if save_fig_dir else None
            try:
                metrics_e = plot_scatter_comparison_ert(model, ert_handler, domain, t, config,
                                                      plot_manager, fname_scatter_e)
                if metrics_e:
                    all_ert_metrics[t] = metrics_e
            except Exception as e:
                logging.error(f"Error creating ERT scatter plot at t={t}: {e}")
            
            # 6. ERT data visualization
            # FIX: Use :06.1f instead of :03d for float values
            fname_ert = os.path.join(save_fig_dir, 'ert_data', f'ert_data_t{hydrus_time:06.1f}.png') if save_fig_dir else None
            try:
                plot_ert_data(ert_handler, hydrus_time, domain, config, plot_manager, fname_ert)
            except Exception as e:
                logging.error(f"Error plotting ERT data at t={hydrus_time}: {e}")
        
        # 7. Cross-section plots
        fname_xsection = os.path.join(save_fig_dir, 'cross_sections', f'cross_sections_t{t:03d}.png') if save_fig_dir else None
        try:
            plot_cross_sections(model, hydrus_handler, ert_handler, domain, t, config,
                              plot_manager, fname_xsection)
        except Exception as e:
            logging.error(f"Error creating cross-section plot at t={t}: {e}")
    
    # 8. Time series plots at monitoring locations
    fname_timeseries = os.path.join(save_fig_dir, 'monitoring_timeseries.png') if save_fig_dir else None
    try:
        plot_monitoring_timeseries(model, hydrus_handler, domain, all_pinn_times,
                                 config, plot_manager, fname_timeseries)
    except Exception as e:
        logging.error(f"Error creating time series plot: {e}")
    
    # 9. Metrics evolution plot
    if all_hydrus_metrics or all_ert_metrics:
        fname_metrics = os.path.join(save_fig_dir, 'metrics_evolution_all_times.png') if save_fig_dir else None
        try:
            create_metrics_summary_plot(all_hydrus_metrics, all_ert_metrics,
                                      plot_manager, fname_metrics)
        except Exception as e:
            logging.error(f"Error creating metrics summary: {e}")
    
    # 10. Save summary statistics
    if save_fig_dir and (all_hydrus_metrics or all_ert_metrics):
        try:
            create_summary_statistics(all_hydrus_metrics, all_ert_metrics, save_fig_dir)
        except Exception as e:
            logging.error(f"Error creating summary statistics: {e}")
    
    logging.info("="*80)
    logging.info(f"COMPREHENSIVE PLOTTING COMPLETE - Created plots for {len(all_pinn_times)} timesteps")
    logging.info("="*80)

def create_summary_statistics(hydrus_metrics: Dict, ert_metrics: Dict,
                            save_fig_dir: str) -> None:
    """Create and save summary statistics"""
    summary_file = os.path.join(save_fig_dir, 'summary_statistics.txt')
    
    with open(summary_file, 'w') as f:
        f.write("PINN MODEL EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        if hydrus_metrics:
            f.write("HYDRUS COMPARISON METRICS:\n")
            f.write("-"*30 + "\n")
            
            times = sorted(hydrus_metrics.keys())
            rmse_vals = [hydrus_metrics[t]['rmse'] for t in times]
            r2_vals = [hydrus_metrics[t]['r2'] for t in times]
            
            f.write(f"Time range: {min(times):.1f} - {max(times):.1f} days\n")
            f.write(f"Average RMSE: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}\n")
            f.write(f"Average R²: {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}\n")
            f.write(f"Best RMSE: {min(rmse_vals):.4f} at t={times[np.argmin(rmse_vals)]:.1f} days\n")
            f.write(f"Worst RMSE: {max(rmse_vals):.4f} at t={times[np.argmax(rmse_vals)]:.1f} days\n\n")
        
        if ert_metrics:
            f.write("ERT COMPARISON METRICS:\n")
            f.write("-"*30 + "\n")
            
            times = sorted(ert_metrics.keys())
            rmse_vals = [ert_metrics[t]['rmse'] for t in times]
            r2_vals = [ert_metrics[t]['r2'] for t in times]
            
            f.write(f"Time points: {len(times)}\n")
            f.write(f"Average RMSE: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}\n")
            f.write(f"Average R²: {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}\n")
            f.write(f"Best RMSE: {min(rmse_vals):.4f} at t={times[np.argmin(rmse_vals)]:.1f} days\n")
            f.write(f"Worst RMSE: {max(rmse_vals):.4f} at t={times[np.argmax(rmse_vals)]:.1f} days\n")
    
    logging.info(f"Summary statistics saved to: {summary_file}")

def plot_results(model: nn.Module, loss_history: Dict, data: Dict, config: Config) -> None:
    """Main function to create all plots"""
    plot_manager = PlotManager(config)
    output_dir = os.path.join(config.training.output_dir, "figures")
    
    # Create comprehensive plots for ALL timesteps
    create_comprehensive_plots(
        model, data['hydrus_handler'], data['ert_handler'], data['hydrus_data'],
        data['hydrus_interp'], data['hydrus_data']['domain'], config, plot_manager, output_dir
    )
    
    # Additionally create specific plots if needed
    # Plot loss history
    plot_loss_history(loss_history, plot_manager,
                     os.path.join(output_dir, "loss_history.png"))
    
    logging.info(f"All plots saved to: {output_dir}")

###############################
# Main Entry Point
###############################

def main():
    """Main function"""
    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    setup_logging(f"pinn_training_with_transpiration_{timestamp}.log")
    
    logging.info("="*80)
    logging.info("IMPROVED PINN WITH COMPREHENSIVE TRANSPIRATION LOSS")
    logging.info("="*80)
    logging.info(f"Device: {device}")
    
    # Create configuration
    config = Config()
    
    # Configure transpiration loss
    config.training.lambda_transpiration = 1.0  # Adjust this weight as needed
    config.training.transpiration_integration_points = 2000  # Higher for better accuracy
    
    # Add timestamp to directories
    config.training.output_dir = f"output_with_transpiration_31_{timestamp}"
    config.training.checkpoint_dir = f"checkpoints_with_transpiration_31_{timestamp}"
    config.training.tensorboard_dir = f"runs_with_transpiration_31_{timestamp}"
    
    logging.info(f"Transpiration loss weight: {config.training.lambda_transpiration}")
    logging.info(f"Integration points: {config.training.transpiration_integration_points}")
    
    # Run training
    try:
        model, loss_history, data = train_pinn_with_adaptive_sampling(config)
        
        # Plot results
        logging.info("\nGenerating plots...")
        plot_results(model, loss_history, data, config)
        
        logging.info("\nTraining completed successfully!")
        logging.info(f"Results saved in: {config.training.output_dir}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

