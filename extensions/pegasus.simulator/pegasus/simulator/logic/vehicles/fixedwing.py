"""
| File: fixedwing.py
| Author: Ahmed Zeer & Mert Colpan
| License: BSD-3-Clause
| Description: Definition of the FixedWing class which is used as the base for all fixed-wing aircraft.
"""

import numpy as np
from pxr import Usd, UsdPhysics, UsdGeom, Gf
from scipy.spatial.transform import Rotation

from omni.isaac.dynamic_control import _dynamic_control

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle import Vehicle

# Mavlink interface (veya kendi backend'iniz)
from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig

# Sensors
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS

# Dynamics
from pegasus.simulator.logic.dynamics import LinearDrag

# Force UI for coordinate debugging
from pegasus.simulator.ui.force_debugger import ForceControlWindow, DebugVisualizer

# Log
import carb
import csv
import os
import time

# Constants for physical limits
MIN_THROTTLE = -500.0
MAX_THROTTLE = 500.0

MIN_FORCE = -1000.0
MAX_FORCE = 1000.0

MIN_MOMENTS = -500.0
MAX_MOMENTS = 500.0

class FixedWingConfig:
    """
    A data class that is used for configuring a Fixed-Wing Aircraft
    """

    def __init__(self):
        """
        Initialization of the FixedWingConfig class
        """

        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "fixedwing"

        # ============================================
        # PROPELLER/MOTOR CONFIGURATION
        # ============================================
        self.prop_max_thrust = 50.0  # Maximum thrust in Newtons
        self.prop_max_rpm = 8000.0   # Maximum RPM
        self.prop_thrust_coefficient = 0.00001  # Thrust = coef * RPM^2
        
        # Motor rotation direction (1: CCW, -1: CW when viewed from behind)
        self.prop_rotation_dir = 1
        self.propeller_joint_name = "propeller_joint"
        self.prop_visual_max_dof_speed = 60.0

        # ============================================
        # AERODYNAMIC COEFFICIENTS
        # ============================================
        # Lift coefficients
        self.CL_0 = 0.28       # Zero angle of attack lift coefficient
        self.CL_alpha = 3.45   # Lift curve slope (per radian)
        self.CL_max = 1.4      # Maximum lift coefficient (stall limit)
        self.CL_min = -1.1     # Minimum lift coefficient
        
        # Drag coefficients
        self.CD_0 = 0.03       # Zero-lift drag coefficient (parasitic drag)
        self.CD_alpha = 0.30   # Drag due to angle of attack
        self.CD_alpha2 = 2.0   # Induced drag factor (quadratic term)
        
        # Side force coefficients
        self.CY_beta = -0.98   # Side force due to sideslip
        
        # Pitch moment coefficients
        self.Cm_0 = -0.02      # Zero angle of attack pitch moment
        self.Cm_alpha = -0.38  # Pitch moment curve slope
        
        # Roll moment coefficients  
        self.Cl_beta = -0.12   # Roll moment due to sideslip (dihedral effect)
        self.Cl_p = -0.50      # Roll damping due to roll rate
        self.Cl_r = 0.15       # Roll-yaw coupling due to yaw rate
        
        # Yaw moment coefficients
        self.Cn_beta = 0.25    # Yaw moment due to sideslip (weathercock stability)
        self.Cn_p = -0.06      # Yaw-roll coupling due to roll rate
        self.Cn_r = -0.20      # Yaw damping due to yaw rate

        # Pitch damping coefficient
        self.Cm_q = -8.0       # Pitch damping due to pitch rate

        # ============================================
        # CONTROL SURFACE DERIVATIVES
        # ============================================
        # Elevator effectiveness (pitch control)
        self.CL_elevator = 0.43    # Lift change per elevator deflection
        self.Cm_elevator = -1.122  # Pitch moment per elevator deflection
        
        # Aileron effectiveness (roll control)
        self.Cl_aileron = 0.229    # Roll moment per aileron deflection
        
        # Rudder effectiveness (yaw control)
        self.Cn_rudder = -0.032    # Yaw moment per rudder deflection
        self.CY_rudder = 0.870     # Side force per rudder deflection

        # ============================================
        # AIRCRAFT GEOMETRY
        # ============================================
        self.wing_area = 0.55          # Wing area in m^2
        self.wing_span = 2.8           # Wing span in m
        self.chord = 0.18880           # Mean aerodynamic chord in m
        self.air_density = 1.225       # Air density in kg/m^3 (sea level)

        # ============================================
        # DRAG CONFIGURATION
        # ============================================
        # Linear drag (for body frame drag effects)
        self.drag = LinearDrag([0.1, 0.1, 0.1])

        # ============================================
        # SENSORS
        # ============================================
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]
        self.graphical_sensors = []
        self.graphs = []

        # ============================================
        # BACKENDS
        # ============================================
        # Control backend (Mavlink, ROS2, or custom)
        self.backends = []

        # ============================================
        # SIMULATION MODE
        # ============================================
        # Modes:
        # - "autonomous": backend throttle/surfaces + thrust + aero + drag + aero moments
        # - "thrust_only": UI thrust + aero + drag + aero moments
        # - "manual": direct UI body-frame force/torque only
        # Backward compatibility: "full" is treated as "autonomous".
        self.simulation_mode = "autonomous"
        self.debug_mode = False
        self.body_name = "/body"
        self.propeller_name = "/propeller"

        # Optional fallback mass for diagnostics when USD mass cannot be read.
        self.mass_kg = 1.0

        # Rate of takeoff viability diagnostics [Hz]
        self.viability_log_rate_hz = 10.0

        # Control input sign mapping (useful when backend and sim conventions differ).
        # Defaults assume backend commands are FRD-like while the simulated body is FLU:
        # roll about X unchanged, pitch and yaw signs flipped.
        self.aileron_sign = 1.0
        self.elevator_sign = -1.0
        self.throttle_sign = 1.0
        self.rudder_sign = -1.0


class FixedWing(Vehicle):
    """
    FixedWing class - Defines a base interface for creating fixed-wing aircraft
    """
    
    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "fixedwing",
        usd_file: str = "",
        vehicle_id: int = 0,
        # Spawning pose of the vehicle
        init_pos=[0.0, 0.0, 0.5],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        config=FixedWingConfig(),
    ):
        """
        Initializes the fixed-wing aircraft object

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator. Defaults to "fixedwing".
            usd_file (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            vehicle_id (int): The id to be used for the vehicle. Defaults to 0.
            init_pos (list): Initial position in inertial frame (ENU). Defaults to [0.0, 0.0, 0.5].
            init_orientation (list): Initial orientation quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (FixedWingConfig): Configuration object. Defaults to FixedWingConfig().
        """

        # 1. Initialize the Vehicle base class
        super().__init__(
            stage_prefix, 
            usd_file, 
            init_pos, 
            init_orientation, 
            config.sensors, 
            config.graphical_sensors, 
            config.graphs, 
            config.backends
        )

        # 2. Store configuration parameters
        self._config = config
        
        # Propeller configuration
        self._prop_max_thrust = config.prop_max_thrust
        self._prop_max_rpm = config.prop_max_rpm
        self._prop_thrust_coef = config.prop_thrust_coefficient
        self._prop_rotation_dir = config.prop_rotation_dir
        self._propeller_joint_name = config.propeller_joint_name
        self._prop_visual_max_dof_speed = config.prop_visual_max_dof_speed
        
        # Aerodynamic coefficients
        self._CL_0 = config.CL_0
        self._CL_alpha = config.CL_alpha
        self._CL_max = config.CL_max
        self._CL_min = config.CL_min
        
        self._CD_0 = config.CD_0
        self._CD_alpha = config.CD_alpha
        self._CD_alpha2 = config.CD_alpha2
        
        self._CY_beta = config.CY_beta
        
        self._Cm_0 = config.Cm_0
        self._Cm_alpha = config.Cm_alpha
        
        self._Cl_beta = config.Cl_beta
        self._Cl_p = config.Cl_p
        self._Cl_r = config.Cl_r
        self._Cn_beta = config.Cn_beta
        self._Cn_p = config.Cn_p
        self._Cn_r = config.Cn_r
        self._Cm_q = config.Cm_q
        
        # Control surface derivatives
        self._CL_elevator = config.CL_elevator
        self._Cm_elevator = config.Cm_elevator
        self._Cl_aileron = config.Cl_aileron
        self._Cn_rudder = config.Cn_rudder
        self._CY_rudder = config.CY_rudder
        
        # Aircraft geometry
        self._wing_area = config.wing_area
        self._wing_span = config.wing_span
        self._chord = config.chord
        self._air_density = config.air_density
        
        # Drag model
        self._drag = config.drag
        self._simulation_mode = "autonomous"
        self.set_simulation_mode(config.simulation_mode)
        self._mass_kg_fallback = float(config.mass_kg)
        self._vehicle_mass_kg = None
        self._viability_log_period = 1.0 / max(float(config.viability_log_rate_hz), 0.1)
        self._viability_log_accum = 0.0
        self._aileron_sign = float(config.aileron_sign)
        self._elevator_sign = float(config.elevator_sign)
        self._throttle_sign = float(config.throttle_sign)
        self._rudder_sign = float(config.rudder_sign)
        self._body_name = getattr(config, 'body_name', "/body")
        self._propeller_name = getattr(config, 'propeller_name', "/propeller")
        
        # Current control inputs (will be updated from backend)
        self._throttle = 0.0      # 0.0 to 1.0
        self._elevator = 0.0      # -1.0 to 1.0 (radians or normalized)
        self._aileron = 0.0       # -1.0 to 1.0
        self._rudder = 0.0        # -1.0 to 1.0
        self._prop_articulation = None
        self._prop_joint_handle = None
        self._prop_joint_resolved = False

        if config.simulation_mode != 'autonomous':
            self.force_ui = ForceControlWindow()

        self.debug_drawer = DebugVisualizer()


        # Initialize CSV logging
        self._log_file_path = "forces_log.csv"
        try:
            self._log_file = open(self._log_file_path, 'w', newline='')
            self._csv_writer = csv.writer(self._log_file)
            # Write header
            self._csv_writer.writerow([
                'timestamp', 'aot', 'sideslip', 'thrust', 'lift', 'drag', 'side',
                'mx', 'my', 'mz', 'u', 'v', 'w', 'roll_deg', 'pitch_deg', 'yaw_deg', 'p', 'q', 'r'
            ])
            self._log_file.flush()
            carb.log_info(f"Initialized forces logging to {self._log_file_path}")
        except Exception as e:
            carb.log_error(f"Failed to initialize forces logging: {e}")
            self._log_file = None
            self._csv_writer = None

    def start(self):
        """
        Called when simulation starts
        """
        pass

    def stop(self):
        """
        Called when simulation stops
        """
        pass

    def set_simulation_mode(self, mode: str):
        """
        Select runtime force model for update():
        - autonomous (or full)
        - thrust_only
        - manual
        """
        valid_modes = {"autonomous", "full", "thrust_only", "manual"}
        selected_mode = str(mode).strip().lower()
        if selected_mode not in valid_modes:
            carb.log_warn(f"Invalid simulation_mode '{mode}', falling back to 'autonomous'.")
            selected_mode = "autonomous"
        self._simulation_mode = selected_mode
        carb.log_info(f"FixedWing simulation mode set to: {self._simulation_mode}")

    def _log_state(self, aot=0, sideslip=0, thrust=0, lift=0, drag=0, side=0, mx=0, my=0, mz=0, u=0, v=0, w=0):
        """
        Logs the vehicle state to the CSV file.
        """
        if not self._csv_writer:
            return

        try:
            timestamp = self._state.time if hasattr(self._state, 'time') else time.time()
            roll_deg, pitch_deg, yaw_deg = Rotation.from_quat(self._state.attitude).as_euler("xyz", degrees=True)
            p, q, r = self._state.angular_velocity
            self._csv_writer.writerow([
                timestamp,
                aot,
                sideslip,
                thrust,
                lift,
                drag,
                side,
                mx,
                my,
                mz,
                u,
                v,
                w,
                roll_deg,
                pitch_deg,
                yaw_deg,
                p,
                q,
                r,
            ])
            self._log_file.flush()
        except Exception as e:
            carb.log_warn(f"Failed to log forces: {e}")

    def _get_vehicle_mass_kg(self):
        """Read rigid-body mass from USD MassAPI, fallback to configured mass."""
        if self._vehicle_mass_kg is not None:
            return self._vehicle_mass_kg

        mass_kg = self._mass_kg_fallback
        try:
            body_prim = self._world.stage.GetPrimAtPath(self._stage_prefix + self._body_name)
            if body_prim and body_prim.IsValid():
                mass_attr = UsdPhysics.MassAPI(body_prim).GetMassAttr()
                if mass_attr and mass_attr.HasAuthoredValue():
                    mass_from_usd = mass_attr.Get()
                    if mass_from_usd is not None and mass_from_usd > 0.0:
                        mass_kg = float(mass_from_usd)
        except Exception as e:
            carb.log_warn(f"Could not read vehicle mass from USD: {e}")

        self._vehicle_mass_kg = mass_kg
        carb.log_info(f"FixedWing mass used for diagnostics: {self._vehicle_mass_kg:.3f} kg")
        return self._vehicle_mass_kg

    def _maybe_log_takeoff_viability(self, dt: float, thrust_force, aero_forces, drag_force):
        """Periodic force-balance diagnostics to understand takeoff feasibility."""
        self._viability_log_accum += dt
        if self._viability_log_accum < self._viability_log_period:
            return
        self._viability_log_accum = 0.0

        mass_kg = self._get_vehicle_mass_kg()
        weight_n = mass_kg * 9.81
        total_force = thrust_force + aero_forces + drag_force
        net_up_n = total_force[2] - weight_n
        drag_back_n = -aero_forces[0]
        lift_up_n = aero_forces[2]

        carb.log_info(
            f"[TakeoffCheck] mass={mass_kg:.3f}kg W={weight_n:.2f}N "
            f"T={thrust_force[0]:.2f}N D={drag_back_n:.2f}N L={lift_up_n:.2f}N "
            f"Fx_net={total_force[0]:.2f}N Fz_net_minus_W={net_up_n:.2f}N"
        )

    def update(self, dt: float):
        """
        Main fixed-wing update callback. Uses the selected simulation mode:
        - autonomous
        - thrust_only
        - manual
        """
        if not self._prim.IsValid():
            return

        pos = self._state.position
        offset_pos = [pos[0], pos[1], pos[2] + 0.2]
        self.debug_drawer.clear()

        V_body = self._state.linear_body_velocity
        u, v, w = V_body[0], V_body[1], V_body[2]
        V = np.linalg.norm(V_body)
        alpha = np.arctan2(-w, u) if V > 0.1 else 0.0
        beta = np.arcsin(np.clip(-v / V, -1.0, 1.0)) if V > 0.1 else 0.0

        mode = self._simulation_mode
        r = Rotation.from_quat(self._state.attitude)

        def _to_world(vec):
            return r.apply(np.asarray(vec, dtype=float)).tolist()

        if mode == "manual":
            forces_body, torques_body = self.force_ui.get_inputs()
            forces_body = np.clip(forces_body, MIN_FORCE, MAX_FORCE)
            torques_body = np.clip(torques_body, MIN_MOMENTS, MAX_MOMENTS)
            self.debug_drawer.draw_vector(pos, _to_world(forces_body), color=(1, 0, 0, 1), scale=0.1)
            self.debug_drawer.draw_vector(offset_pos, _to_world(torques_body), color=(0, 0, 1, 1), scale=0.1)

            # Apply at CoM offset to avoid parasitic yaw from off-center body origin
            _offset = getattr(self._config, 'force_application_offset', np.array([0.0, 0.0, 0.0]))
            self.apply_force(forces_body, pos=_offset, body_part=self._body_name)
            self.apply_torque(torques_body, body_part=self._body_name)

            self._log_state(
                aot=alpha,
                sideslip=beta,
                thrust=forces_body[0],
                lift=forces_body[2],
                drag=-forces_body[0],
                side=forces_body[1],
                mx=torques_body[0],
                my=torques_body[1],
                mz=torques_body[2],
                u=u,
                v=v,
                w=w
            )

        elif mode == "thrust_only":
            # UI drives thrust on X and optional manual pitch torque on Y.
            # Aerodynamic forces/moments are still active.
            forces_body, torques_body = self.force_ui.get_inputs()
            thrust_force = np.array([forces_body[0], 0.0, 0.0])
            thrust_force = np.clip(thrust_force, MIN_THROTTLE, MAX_THROTTLE)
            manual_pitch_torque = np.array([0.0, torques_body[1], 0.0])
            manual_pitch_torque = np.clip(manual_pitch_torque, MIN_MOMENTS, MAX_MOMENTS)

            data = self._calculate_aerodynamics()
            aero_forces = np.clip(data['forces'], MIN_FORCE, MAX_FORCE)
            aero_moments = np.clip(data['moments'], MIN_MOMENTS, MAX_MOMENTS)
            total_moments = np.clip(aero_moments + manual_pitch_torque, MIN_MOMENTS, MAX_MOMENTS)
            drag_force = np.clip(self._drag.update(self._state, dt), MIN_FORCE, MAX_FORCE)

            _offset = getattr(self._config, 'force_application_offset', np.array([0.0, 0.0, 0.0]))

            # Convert body-frame CoM offset to world frame for correct arrow origin
            r_world = Rotation.from_quat(self._state.attitude)
            com_world = pos + r_world.apply(_offset)
            com_vis = [com_world[0], com_world[1], com_world[2] + 0.05]  # slight z lift for visibility

            self.debug_drawer.draw_vector(com_vis, _to_world(thrust_force), color=(1, 0, 0, 1), scale=0.1)
            self.debug_drawer.draw_vector(com_vis, _to_world(aero_forces), color=(1, 1, 0, 1), scale=0.1)
            self.debug_drawer.draw_vector(com_vis, _to_world(total_moments), color=(0, 0, 1, 1), scale=0.1)
            self.debug_drawer.draw_vector(com_vis, _to_world(drag_force), color=(0, 1, 1, 1), scale=0.1)

            # Apply forces at CoM to avoid parasitic pitch/yaw torques
            self.apply_force(thrust_force, pos=_offset, body_part=self._body_name)
            self.apply_force(aero_forces, pos=_offset, body_part=self._body_name)
            self.apply_force(drag_force, pos=_offset, body_part=self._body_name)
            self.apply_torque(total_moments, body_part=self._body_name)
            self._update_propeller_visual(thrust_force[0])
            self._maybe_log_takeoff_viability(dt, thrust_force, aero_forces, drag_force)

            self._log_state(
                aot=data['aot'],
                sideslip=data['sideslip'],
                thrust=thrust_force[0],
                lift=aero_forces[2],
                drag=-aero_forces[0],
                side=aero_forces[1],
                mx=total_moments[0],
                my=total_moments[1],
                mz=total_moments[2],
                u=u,
                v=v,
                w=w
            )

        else:
            # autonomous mode (backend-driven controls)
            self._update_control_inputs()
            thrust_force = np.clip(self._calculate_propeller_thrust(), MIN_THROTTLE, MAX_THROTTLE)

            data = self._calculate_aerodynamics()
            aero_forces = np.clip(data['forces'], MIN_FORCE, MAX_FORCE)
            aero_moments = np.clip(data['moments'], MIN_MOMENTS, MAX_MOMENTS)
            drag_force = np.clip(self._drag.update(self._state, dt), MIN_FORCE, MAX_FORCE)

            # Activate draw vectors only in debug mode for fully autonomous functionality.
            if self._config.debug_mode:
                self.debug_drawer.draw_vector(pos, _to_world(thrust_force), color=(1, 0, 0, 1), scale=0.1)
                self.debug_drawer.draw_vector(pos, _to_world(aero_forces), color=(1, 1, 0, 1), scale=0.1)
                self.debug_drawer.draw_vector(offset_pos, _to_world(aero_moments), color=(0, 0, 1, 1), scale=0.1)
                self.debug_drawer.draw_vector(offset_pos, _to_world(drag_force), color=(0, 1, 1, 1), scale=0.1)

            # Offset for force application (to handle off-center CAD origins)
            offset_pos = getattr(self._config, 'force_application_offset', np.array([0.0, 0.0, 0.0]))

            self.apply_force(thrust_force, pos=offset_pos, body_part=self._body_name)
            self.apply_force(aero_forces, pos=offset_pos, body_part=self._body_name)
            self.apply_force(drag_force, pos=offset_pos, body_part=self._body_name)
            self.apply_torque(aero_moments, body_part=self._body_name)
            self._update_propeller_visual(thrust_force[0])
            self._maybe_log_takeoff_viability(dt, thrust_force, aero_forces, drag_force)

            self._log_state(
                aot=data['aot'],
                sideslip=data['sideslip'],
                thrust=thrust_force[0],
                lift=aero_forces[2],
                drag=-aero_forces[0],
                side=aero_forces[1],
                mx=aero_moments[0],
                my=aero_moments[1],
                mz=aero_moments[2],
                u=u,
                v=v,
                w=w
            )

        for backend in self._backends:
            backend.update(dt)

    def _update_control_inputs(self):
        if len(self._backends) == 0:
            carb.log_warn("No backend detected @ fixedwing. Control inputs set to 0.")
            self._throttle = 0.0
            self._elevator = 0.0
            self._aileron = 0.0
            self._rudder = 0.0
            return

        backend = self._backends[0]
        raw_inputs = backend.input_reference()
        if raw_inputs is None or len(raw_inputs) < 4:
            carb.log_warn("Backend input_reference has insufficient channels. Expected at least 4.")
            self._throttle = 0.0
            self._elevator = 0.0
            self._aileron = 0.0
            self._rudder = 0.0
            return

        raw_inputs = np.asarray(raw_inputs, dtype=float)

        # NOTA: No bloqueamos por backend._armed porque el GCS (udp:14550) puede
        # consumir los HEARTBEAT antes de que el backend los reciba.
        # Si ArduPilot está desarmado, los servos que envía son neutros/cero — lo
        # detectamos abajo al calcular las superficies. No necesitamos el flag aquí.

        # Decode ArduPilot/PX4 style scaled outputs when ThrusterControl metadata is available.
        has_rotor_data = hasattr(backend, "_rotor_data")
        if has_rotor_data:
            tc = backend._rotor_data

            def _recover_fraction(value, channel_idx):
                scaling = tc.input_scaling[channel_idx] if channel_idx < len(tc.input_scaling) else 1.0
                if scaling == 0.0:
                    scaling = 1.0
                zero = tc.zero_position_armed[channel_idx] if channel_idx < len(tc.zero_position_armed) else 0.0
                offset = tc.input_offset[channel_idx] if channel_idx < len(tc.input_offset) else 0.0
                return (value - zero) / scaling - offset

            ail_raw = _recover_fraction(raw_inputs[0], 0)
            ele_raw = _recover_fraction(raw_inputs[1], 1)
            thr_raw = _recover_fraction(raw_inputs[2], 2)
            rud_raw = _recover_fraction(raw_inputs[3], 3)
        else:
            ail_raw, ele_raw, thr_raw, rud_raw = raw_inputs[:4]

        def _normalize_surface(x):
            if has_rotor_data:
                # ArduPilot decoded values are fractions in [0, 1], center at 0.5.
                return (x - 0.5) * 2.0
            # Generic backends may publish either [-1,1] or [0,1].
            if 0.0 <= x <= 1.0:
                return (x - 0.5) * 2.0
            return x

        def _normalize_throttle(x):
            if has_rotor_data:
                return x
            if 0.0 <= x <= 1.0:
                return x
            if -1.0 <= x <= 1.0:
                return 0.5 * (x + 1.0)
            return x

        self._aileron = np.clip(self._aileron_sign * _normalize_surface(ail_raw), -1.0, 1.0)
        self._elevator = np.clip(self._elevator_sign * _normalize_surface(ele_raw), -1.0, 1.0)
        self._throttle = np.clip(self._throttle_sign * _normalize_throttle(thr_raw), 0.0, 1.0)
        self._rudder = np.clip(self._rudder_sign * _normalize_surface(rud_raw), -1.0, 1.0)

        carb.log_info(
            f"FixedWing Inputs -> Ail: {self._aileron:.3f}, "
            f"Ele: {self._elevator:.3f}, "
            f"Thr: {self._throttle:.3f}, "
            f"Rud: {self._rudder:.3f} | "
            f"signs(A,E,T,R)=({self._aileron_sign:+.0f},{self._elevator_sign:+.0f},{self._throttle_sign:+.0f},{self._rudder_sign:+.0f})"
        )

    def _calculate_propeller_thrust(self) -> float:
        """
        Calculates thrust force from propeller based on throttle input
        
        Returns:
            float: Thrust force in Newtons (body frame X-axis)
        """
        # RPM from throttle
        rpm = self._throttle * self._prop_max_rpm
        
        # Thrust = coefficient * RPM^2
        thrust = self._prop_thrust_coef * (rpm ** 2)
        
        # Limit to maximum thrust
        thrust = np.clip(thrust, 0.0, self._prop_max_thrust)
        
        return np.array([thrust, 0, 0])

    def _calculate_aerodynamics(self):
        """
        Calculates aerodynamic forces and moments based on current flight state
        
        Returns:
            tuple: (forces, moments) in body frame
                forces: [Fx, Fy, Fz] in Newtons
                moments: [Mx, My, Mz] in Newton-meters
        """
        
        # Get airspeed in body frame [u, v, w]
        V_body = self._state.linear_body_velocity
        u, v, w = V_body[0], V_body[1], V_body[2]
        
        # Total airspeed
        V = np.sqrt(u*u + v*v + w*w)

        carb.log_info(f"\n\n--- Aerodynamics Cycle Start ---")
        carb.log_info(f"Body Vel: {V_body}")
        carb.log_info(f"Aero Vel: [u={u:.4f}, v={v:.4f}, w={w:.4f}], Total V={V:.4f}")
        
        if V < 0.1:
            carb.log_info("Speed too low (< 0.1), returning zero forces/moments.")
            return {
                'forces': np.array([0.0, 0.0, 0.0]),
                'moments': np.array([0.0, 0.0, 0.0]),
                'aot': 0.0,
                'sideslip': 0.0
            }
        
        # Convert FLU body velocity to FRD-compatible aerodynamic angles:
        # FRD uses +Y right, +Z down; FLU uses +Y left, +Z up.
        alpha = np.arctan2(-w, u)  # radians
        beta = np.arcsin(np.clip(-v / V, -1.0, 1.0))  # radians

        carb.log_info(f"Angles: Alpha={np.degrees(alpha):.4f} deg ({alpha:.4f} rad), "
                    f"Beta={np.degrees(beta):.4f} deg ({beta:.4f} rad)")
        
        # Dynamic pressure
        q = 0.5 * self._air_density * V**2
        carb.log_info(f"Dynamic Pressure (q): {q:.4f} Pa (rho={self._air_density})")

        # Body angular rates are in FLU. Convert to FRD-compatible rates used by
        # conventional aerodynamic derivatives (p same, q/r sign inverted).
        p_flu, q_flu, r_flu = self._state.angular_velocity
        p_rate = p_flu
        q_rate = -q_flu
        r_rate = -r_flu

        p_hat = p_rate * self._wing_span / (2.0 * V)
        q_hat = q_rate * self._chord / (2.0 * V)
        r_hat = r_rate * self._wing_span / (2.0 * V)
        
        # ==========================================
        # LIFT COEFFICIENT
        # ==========================================
        CL_raw = self._CL_0 + self._CL_alpha * alpha + self._CL_elevator * self._elevator
        CL = np.clip(CL_raw, self._CL_min, self._CL_max)  # Stall limits
        carb.log_info(f"Coeff Lift (CL): {CL:.4f} (Raw: {CL_raw:.4f}, Control: {self._elevator})")
        
        # ==========================================
        # DRAG COEFFICIENT
        # ==========================================
        CD = self._CD_0 + self._CD_alpha * abs(alpha) + self._CD_alpha2 * alpha**2
        carb.log_info(f"Coeff Drag (CD): {CD:.4f}")
        
        # ==========================================
        # SIDE FORCE COEFFICIENT
        # ==========================================
        CY = self._CY_beta * beta + self._CY_rudder * self._rudder
        carb.log_info(f"Coeff Side (CY): {CY:.4f} (Control: {self._rudder})")
        
        # ==========================================
        # MOMENT COEFFICIENTS
        # ==========================================
        # Roll moment (around X-axis)
        Cl = (
            self._Cl_beta * beta
            + self._Cl_aileron * self._aileron
            + self._Cl_p * p_hat
            + self._Cl_r * r_hat
        )
        
        # Pitch moment (around Y-axis)
        Cm = (
            self._Cm_0
            + self._Cm_alpha * alpha
            + self._Cm_elevator * self._elevator
            + self._Cm_q * q_hat
        )
        
        # Yaw moment (around Z-axis)
        Cn = (
            self._Cn_beta * beta
            + self._Cn_rudder * self._rudder
            + self._Cn_p * p_hat
            + self._Cn_r * r_hat
        )

        carb.log_info(f"Moment Coeffs: Cl (Roll)={Cl:.4f}, Cm (Pitch)={Cm:.4f}, Cn (Yaw)={Cn:.4f}")
        
        # ==========================================
        # FORCES (Stability frame -> Aero Body frame)
        # ==========================================
        # In stability frame:
        # -D (drag) along velocity
        # Y (side force) perpendicular to velocity and lift
        # -L (lift) perpendicular to velocity and side force
        
        L = CL * q * self._wing_area
        D = CD * q * self._wing_area
        Y = CY * q * self._wing_area
        
        carb.log_info(f"Stability Forces: Lift={L:.4f} N, Drag={D:.4f} N, Side={Y:.4f} N")

        # Transform from stability to Aero Body frame
        # Stability frame is rotated by alpha around Y-axis relative to Aero Body
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        Fx_a = -D * cos_alpha + L * sin_alpha
        Fy_a = Y
        Fz_a = -D * sin_alpha - L * cos_alpha
        
        forces_aero = np.array([Fx_a, Fy_a, Fz_a])
        carb.log_info(f"Aero Body Forces: Fx={Fx_a:.4f}, Fy={Fy_a:.4f}, Fz={Fz_a:.4f}")
        
        # ==========================================
        # MOMENTS
        # ==========================================
        Mx_a = Cl * q * self._wing_area * self._wing_span
        My_a = Cm * q * self._wing_area * self._chord
        Mz_a = Cn * q * self._wing_area * self._wing_span
        
        moments_aero = np.array([Mx_a, My_a, Mz_a])
        carb.log_info(f"Aero Body Moments: Mx={Mx_a:.4f}, My={My_a:.4f}, Mz={Mz_a:.4f}")

        carb.log_info(f"--- Aerodynamics Cycle Ends ---\n\n")
        
        forces_flu = np.array([
             forces_aero[0],   # Fx (Forward) unchanged
            -forces_aero[1],   # Fy: Right → Left
            -forces_aero[2]    # Fz: Down → Up
        ])

        moments_flu = np.array([
             moments_aero[0],  # Mx (Roll) unchanged
            -moments_aero[1],  # My (Pitch) sign flipped
            -moments_aero[2]   # Mz (Yaw) sign flipped
        ])

        return dict(forces=forces_flu, moments=moments_flu, aot=alpha, sideslip=beta)

    def _resolve_propeller_joint(self):
        """Resolve and cache the DOF handle used for propeller visual spin."""
        if self._prop_joint_resolved:
            return

        # Try to get the articulation from the root prim (where ArticulationRootAPI lives)
        articulation = None
        # First, try the body_name's parent (the root xform of the imported model)
        body_name = getattr(self, '_body_name', '/body')
        parts = body_name.strip('/').split('/')
        if len(parts) >= 1:
            root_prim_path = self._stage_prefix + '/' + parts[0]
        else:
            root_prim_path = self._stage_prefix + self._propeller_name

        for candidate_path in [root_prim_path, self._stage_prefix + self._propeller_name, self._stage_prefix]:
            try:
                articulation = self.get_dc_interface().get_articulation(candidate_path)
                if articulation is not None:
                    carb.log_info(f"FixedWing: Found articulation at '{candidate_path}'")
                    break
            except RuntimeError:
                carb.log_info(f"FixedWing: No articulation at '{candidate_path}', trying next...")
                continue

        if articulation is None:
            self._prop_joint_resolved = True
            carb.log_warn(f"FixedWing: Could not find articulation for propeller. Skipping visual spin.")
            return

        try:
            # DOF lookup expects the joint dof token name.
            joint = self.get_dc_interface().find_articulation_dof(articulation, self._propeller_joint_name)
            invalid_handle = getattr(_dynamic_control, "INVALID_HANDLE", -1)
            if joint is not None and joint != invalid_handle:
                self._prop_articulation = articulation
                self._prop_joint_handle = joint
                self._prop_joint_resolved = True
                carb.log_info(f"FixedWing propeller visual joint resolved: {self._propeller_joint_name}")
                return
            self._prop_joint_resolved = True
            carb.log_warn(f"FixedWing propeller visual joint '{self._propeller_joint_name}' not found under '{self._stage_prefix}'. Skipping visual spin.")
        except Exception as e:
            self._prop_joint_resolved = True
            carb.log_warn(f"Failed resolving fixed-wing propeller joint '{self._propeller_joint_name}': {e}")

    def _update_propeller_visual(self, thrust_n: float = None):
        """
        Updates the propeller joint velocity for visual animation
        """
        self._resolve_propeller_joint()
        if self._prop_joint_handle is None:
            return

        throttle_frac = np.clip(self._throttle, 0.0, 1.0)
        if thrust_n is not None and self._prop_max_thrust > 1e-6:
            throttle_frac = max(throttle_frac, float(np.clip(thrust_n / self._prop_max_thrust, 0.0, 1.0)))

        if throttle_frac <= 0.01:
            target_speed = 0.0
        else:
            min_spin = 5.0
            target_speed = min_spin + throttle_frac * max(self._prop_visual_max_dof_speed - min_spin, 0.0)
            target_speed *= np.sign(self._prop_rotation_dir) if self._prop_rotation_dir != 0 else 1.0

        try:
            self.get_dc_interface().set_dof_velocity(self._prop_joint_handle, target_speed)
        except Exception as e:
            carb.log_warn(f"Failed to set fixed-wing propeller visual speed: {e}")


    def set_control_inputs(self, throttle=None, elevator=None, aileron=None, rudder=None):
        """
        Manuel olarak kontrol girişlerini ayarlamak için yardımcı fonksiyon
        (Test amaçlı kullanılabilir)
        
        Args:
            throttle (float): 0.0 to 1.0
            elevator (float): -1.0 to 1.0
            aileron (float): -1.0 to 1.0
            rudder (float): -1.0 to 1.0
        """
        if throttle is not None:
            self._throttle = np.clip(throttle, 0.0, 1.0)
        if elevator is not None:
            self._elevator = np.clip(elevator, -1.0, 1.0)
        if aileron is not None:
            self._aileron = np.clip(aileron, -1.0, 1.0)
        if rudder is not None:
            self._rudder = np.clip(rudder, -1.0, 1.0)

    def get_aerodynamic_state(self):
        """
        Returns current aerodynamic state information
        Debugging ve analiz için kullanışlı
        
        Returns:
            dict: Dictionary containing aerodynamic parameters
        """
        V_body = self._state.linear_body_velocity
        u, v, w = V_body[0], V_body[1], V_body[2]
        V = np.linalg.norm(V_body)
        
        alpha = np.arctan2(-w, u) if V > 0.1 else 0.0
        beta = np.arcsin(np.clip(-v / V, -1.0, 1.0)) if V > 0.1 else 0.0
        
        return {
            'airspeed': V,
            'angle_of_attack': np.degrees(alpha),
            'sideslip': np.degrees(beta),
            'throttle': self._throttle,
            'elevator': self._elevator,
            'aileron': self._aileron,
            'rudder': self._rudder
        }

    def __del__(self):
        """
        Destructor - closes the log file
        """
        try:
            if hasattr(self, '_log_file') and self._log_file:
                self._log_file.close()
                carb.log_info("Closed forces log file.")
        except:
            pass
