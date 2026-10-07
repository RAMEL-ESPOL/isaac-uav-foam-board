import omni.ui as ui
import omni.kit.app
import numpy as np

# 1. Enable the extension dynamically
manager = omni.kit.app.get_app().get_extension_manager()
if not manager.is_extension_enabled("omni.isaac.debug_draw"):
    manager.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
from omni.isaac.debug_draw import _debug_draw

import carb

class ForceControlWindow:
    def __init__(self, title="Force Control", width=300, height=400):
        self._window = ui.Window(title, width=width, height=height)
        
        # Models to store the values (initialized to 0.0)
        self.models = {
            "fx": ui.SimpleFloatModel(0.0),
            "fy": ui.SimpleFloatModel(0.0),
            "fz": ui.SimpleFloatModel(0.0),
            "tx": ui.SimpleFloatModel(0.0),
            "ty": ui.SimpleFloatModel(0.0),
            "tz": ui.SimpleFloatModel(0.0),
        }

        # Build the UI immediately
        self._build_ui()

    def _build_ui(self):
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                
                # --- Linear Forces Section ---
                ui.Label("Linear Forces", height=20, style={"font_size": 16, "color": 0xFFDDDDDD})
                with ui.VStack(spacing=2):
                    self._create_slider("Force X", self.models["fx"], -100.0, 100.0)
                    self._create_slider("Force Y", self.models["fy"], -100.0, 100.0)
                    self._create_slider("Force Z", self.models["fz"], -100.0, 100.0)

                ui.Spacer(height=10)
                ui.Separator(height=2)
                ui.Spacer(height=10)

                # --- Torques Section ---
                ui.Label("Torques", height=20, style={"font_size": 16, "color": 0xFFDDDDDD})
                with ui.VStack(spacing=2):
                    self._create_slider("Torque X (Roll)", self.models["tx"], -50.0, 50.0)
                    self._create_slider("Torque Y (Pitch)", self.models["ty"], -15.0, 15.0, step=0.01)
                    self._create_slider("Torque Z (Yaw)",  self.models["tz"], -50.0, 50.0)

                ui.Spacer(height=15)
                
                # --- Reset Button ---
                ui.Button("ZERO ALL", height=40, clicked_fn=self.zero_all, 
                          style={"background_color": 0xFF5555AA, "font_size": 14})

    def _create_slider(self, label, model, min_val, max_val, step=0.1):
        """Helper to create a consistent label + float drag row + reset button"""
        with ui.HStack(height=24, spacing=5):
            ui.Label(label, width=120, style={"color": 0xFFAAAAAA})
            # FloatDrag with custom step for precision
            ui.FloatDrag(model, min=min_val, max=max_val, step=step)
            # Individual reset button
            ui.Button("0", width=25, clicked_fn=lambda: self.reset_model(model),
                      style={"background_color": 0xFF777777, "font_size": 10})

    def reset_model(self, model):
        """Resets a single model to 0"""
        model.as_float = 0.0

    def zero_all(self):
        """Resets all forces to 0"""
        for model in self.models.values():
            model.as_float = 0.0

    def get_inputs(self):
        """Returns ((fx, fy, fz), (tx, ty, tz))"""
        # Retrieve current values from the models
        f = [self.models["fx"].as_float, 
             self.models["fy"].as_float, 
             self.models["fz"].as_float]
             
        t = [self.models["tx"].as_float, 
             self.models["ty"].as_float, 
             self.models["tz"].as_float]
        return np.array([f, t])

    def shutdown(self):
        """Clean up window"""
        self._window = None
    

class DebugVisualizer:
    def __init__(self):
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def draw_vector(self, origin, vector, color=(1, 0, 0, 1), scale=1.0):
        """
        Draws a line representing a vector (force/torque).
        :param origin: [x, y, z] start point
        :param vector: [x, y, z] direction and magnitude
        :param color: (r, g, b, a) normalized
        :param scale: Visual scaling factor (e.g., if force is 100N, scale by 0.01 to draw a 1m line)
        """
        start = carb.Float3(origin[0], origin[1], origin[2])
        end = carb.Float3(
            origin[0] + vector[0] * scale,
            origin[1] + vector[1] * scale,
            origin[2] + vector[2] * scale
        )
        
        # Draw the main line
        # draw_lines takes lists of start/end points to support batching
        self._draw.draw_lines([start], [end], [color], [5.0]) # 2.0 is thickness

    def clear(self):
        self._draw.clear_lines()