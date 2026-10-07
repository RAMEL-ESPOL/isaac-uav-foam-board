# Isaac UAV

**Isaac UAV** is a fixed-wing UAV simulation stack built on [Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) and [Pegasus Simulator](https://pegasussimulator.github.io/PegasusSimulator/).  
It supports aerodynamic force and moment simulation, interactive force debugging, and backend-driven flight through ArduPilot. [Checkout the Demo.](https://youtu.be/x342-TC76Sw?si=QGYIJR8t6J6wmSoF)

[![DEMO](docs/_static/banner2.png)](https://youtu.be/x342-TC76Sw?si=QGYIJR8t6J6wmSoF)

<table style="width: 100%; text-align: center;">
  <tr>
    <td><strong>Frame Debugging</strong></td>
    <td><strong>Aero Debugging</strong></td>
    <td><strong>Fully Autonomous</strong></td>
  </tr>
  <tr>
    <td><img src="docs/_static/frame_debugger_notext.gif" width="300pt" /></td>
    <td><img src="docs/_static/aero_debugger_2.gif" width="300pt" /></td>
    <td><img src="docs/_static/ardu_demo_1.gif" width="300pt" /></td>
  </tr>
</table>


These three workflows map directly to the fixed-wing modes implemented in this fork:

- `Frame Debugging` (`manual` mode): applies only UI force/torque inputs to validate body-frame signs and axes.
- `Aero Debugging` (`thrust_only` mode): applies user thrust while still computing and applying aerodynamic forces/moments.
- `Fully Autonomous` (`autonomous` mode): reads control channels from ArduPilot backend and runs full aero + propulsion dynamics.

## 12_ardupilot_fixedwing Overview

The `examples/12_ardupilot_fixedwing.py` script is the reference entry point for backend-driven fixed-wing simulation.  
It sets vehicle parameters, selects the simulation mode, and launches ArduPilot integration.

### 1) Configure flight mode and aerodynamic model

```python
config = FixedWingConfig()

config.prop_max_thrust = 100.0
config.prop_max_rpm = 10000.0
config.prop_thrust_coefficient = 0.000075

config.wing_area = 2.36
config.wing_span = 4.46
config.chord = 0.53

config.CL_0 = 0.3
config.CL_alpha = 4.0
config.CL_max = 1.5
config.CD_0 = 0.025

config.simulation_mode = "autonomous"  # autonomous | thrust_only | manual
```

### 2) Enable ArduPilot backend

```python
ardupilot_config = ArduPilotMavlinkBackendConfig({
    "vehicle_id": 0,
    "ardupilot_autolaunch": True,
    "ardupilot_dir": self.pg.ardupilot_path,
    "ardupilot_vehicle_model": "plane",
    "ardupilot_vehicle": "ArduPlane",
})

config.backends = [
    ArduPilotMavlinkBackend(config=ardupilot_config),
]
```

### 3) Spawn the vehicle

```python
self.aircraft = FixedWing(
    stage_prefix="/World/fixedwing0",
    usd_file=ROBOTS["Fixed Wing"],
    vehicle_id=0,
    init_pos=[0.0, 0.0, 1.0],
    init_orientation=Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
    config=config,
)
```

## Foamboard Fixed-Wing Digital Twin (YOY Trainer)

[`examples/yoy_trainer_v2`](examples/yoy_trainer_v2/) is the digital twin of a
low-cost foamboard fixed-wing aircraft (JoyPlanes YOY Trainer) used in the paper
*Integration and Validation of Open-Source Fixed-Wing UAV Platforms for Low-Cost
Autonomous Flight Research* (IEEE ETCM 2026). Its aerodynamic coefficients come
from a CFD characterization of the airframe, and it flies closed-loop against
ArduPilot SITL.

```bash
# Terminal 1 - Isaac Sim + ArduPilot SITL
isaac_run examples/yoy_trainer_v2/14_yoy_trainer_v2_fixedwing.py --mode autonomous

# Terminal 2 - once SITL reports a heartbeat: sustained loiter, 150 m radius
python3 examples/yoy_trainer_v2/scripts/autonomous_flight.py --loiter-now --duration 300 --radius 150
```

The example's [README](examples/yoy_trainer_v2/README.md) lists the aircraft
parameters, the mission flags and the flight-report tooling. CAD, autopilot
parameters, CFD data and the flight logs behind the paper are published in
[RAMEL-ESPOL/OpenSourceFixedWingUAV](https://github.com/RAMEL-ESPOL/OpenSourceFixedWingUAV).

## Latest Updates
- Added the YOY Trainer foamboard digital twin (`examples/yoy_trainer_v2`).
- Patched the fixed-wing aerodynamic model, GPS sensor, ArduPilot backend and force debugger.
- Refined fixed-wing aerodynamics and integrated ArduPilot control flow (`d799373`).
- Added manual flight/debug mode for direct force and moment control (`3b23ff3`).
- Added decoupled force debugging tools and a tracking camera workflow (`a09c34f`).
- Improved visual debugging overlays and markers (`7fb7767`, `1405c07`).
- Added propeller animation support and cleanup updates (`e0552ab`, `6c76a8c`).


## Main Developer Team

Pegasus Simulator is an open-source project initiated by **Marcelo Jacinto** in January 2023.  
This repository is a fixed-wing-focused fork that builds on the original Pegasus architecture and credits the original authors and contributors below.

* Project Founder
	* [Marcelo Jacinto](https://github.com/MarceloJacinto), under the supervision of <u>Prof. Rita Cunha</u> and <u>Prof. Antonio Pascoal</u> (IST/ISR-Lisbon)
* Architecture
  * [Marcelo Jacinto](https://github.com/MarceloJacinto)
  * [João Pinto](https://github.com/jschpinto)
* Multirotor Dynamic Simulation and Control
  * [Marcelo Jacinto](https://github.com/MarceloJacinto)
* Example Applications
	* [Marcelo Jacinto](https://github.com/MarceloJacinto)
	* [João Pinto](https://github.com/jschpinto)
* Ardupilot Integration (Experimental)
  * [Tomer Tiplitsky](https://github.com/TomerTip)
  * [Tanner Gilbert](https://github.com/TannerGilbert)
  * [Seunghwan Jo](https://github.com/SwiftGust)
* Fixed-Wing UAV Integration
  * [Ahmed Zeer](https://github.com/AhmedZeer)

## Citation

If you find Pegasus Simulator useful in your academic work, please cite the paper below. It is also available [here](https://doi.org/10.1109/ICUAS60882.2024.10556959).
```
@INPROCEEDINGS{10556959,
  author={Jacinto, Marcelo and Pinto, João and Patrikar, Jay and Keller, John and Cunha, Rita and Scherer, Sebastian and Pascoal, António},
  booktitle={2024 International Conference on Unmanned Aircraft Systems (ICUAS)}, 
  title={Pegasus Simulator: An Isaac Sim Framework for Multiple Aerial Vehicles Simulation}, 
  year={2024},
  volume={},
  number={},
  pages={917-922},
  keywords={Simulation;Robot sensing systems;Real-time systems;Sensor systems;Sensors;Task analysis},
  doi={10.1109/ICUAS60882.2024.10556959}}
```
