#!/usr/bin/env python
"""
| File: autonomous_flight.py
| Author: Steven Martinez Jara
| License: BSD-3-Clause
| Description: Autonomous waypoint mission + LOITER monitor for the YOY Trainer v2.
|              Connects to ArduPilot SITL via MAVLink, arms in FBWA, takes off,
|              flies WP0 → WP1 → LOITER, and logs telemetry to CSV.
|
| Prerequisites (run both from the repo root):
|   Terminal 1:  isaac_run examples/yoy_trainer_v2/14_yoy_trainer_v2_fixedwing.py --mode autonomous
|   Terminal 2:  python3   examples/yoy_trainer_v2/scripts/autonomous_flight.py
"""

import sys
import time
import csv
import os
import math
import json
import gzip
import shutil

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink is not installed.  pip install pymavlink")
    sys.exit(1)

# ── Paths (relative to this script) ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(SCRIPT_DIR, os.pardir, "flight_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Waypoints (Mission Planner, default ArduPilot SITL home) ─────────────────
HOME   = (-35.360983,   149.166699)
WP0    = (-35.36100000, 149.16900000)   # East  — straight
WP1    = (-35.36327729, 149.16660000)   # South — turn
LOITER = (-35.36293680, 149.16507765)   # Circles west of home

STRAIGHT_WP   = (-35.360983, 149.166699 + 0.5)  # ~45 km due east — never
                                                 # reached, forces sustained
                                                 # straight flight for the
                                                 # --straight diagnostic mode

FLIGHT_ALT    = 50.0    # m AGL
CLIMB_ALT     = 40.0    # m AGL — target for the --loiter-now climb leg.
                         # Lower than FLIGHT_ALT on purpose: the climb is
                         # what runs the speed away (TECS opens the throttle,
                         # the aircraft accelerates past 25 m/s and the roll
                         # axis diverges), so ask for the least height that
                         # still makes the orbit a real airborne one.
LOITER_DIR    = +1      # +1 = clockwise (right turns), -1 = counter-clockwise.
                         # v1 logs had right banks producing almost no turn
                         # (actual / g*tan(phi)/V ratio ~0), so the orbit was
                         # forced counter-clockwise. That was the 57 deg
                         # control surfaces and 15 %-saturated thrust, not a
                         # handedness bug: in auto_20260923_232856 a +21 deg
                         # right bank turned at 0.80 of theory and a -25 deg
                         # left bank at 0.84. Clockwise now, to confirm right
                         # turns in a sustained orbit.
LOITER_RADIUS = 80.0    # m  — was 150 while the aircraft flew at 29 m/s (min
                         # radius ~126 m at 35 deg of bank). At the 16 m/s it
                         # holds now, 80 m needs ~18 deg of bank, half of
                         # ROLL_LIMIT_DEG; the minimum at that limit is ~37 m.
                         # The L1 circle-tracking PD does not depend on
                         # NAVL1_PERIOD's lookahead distance, so a radius
                         # below it (~95 m at period 25) still tracks.
R_EARTH       = 6_371_000.0


# ── Helpers ──────────────────────────────────────────────────────────────────
def dist_m(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * math.cos(math.radians(lat1)) * math.pi / 180 * R_EARTH
    dy = (lat2 - lat1) * math.pi / 180 * R_EARTH
    return math.sqrt(dx * dx + dy * dy)


def connect(uri="tcp:127.0.0.1:5762", retry_timeout_s=90):
    print("Connecting to ArduPilot …")
    deadline = time.time() + retry_timeout_s
    m = None
    while m is None:
        try:
            m = mavutil.mavlink_connection(uri)
        except ConnectionRefusedError:
            if time.time() > deadline:
                print(f"\nERROR: still refused after {retry_timeout_s}s.")
                print("Is Terminal 1 (isaac_run ... --mode autonomous) up and showing 'Detected vehicle' / a heartbeat yet?")
                raise
            print("  ArduPilot SITL not listening yet, retrying...")
            time.sleep(2)
    while True:
        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=15)
        if msg and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
            m.target_system    = msg.get_srcSystem()
            m.target_component = msg.get_srcComponent()
            break
    print("Connected!\n")
    return m


def set_params(m, params):
    """Set parameters and VERIFY each one was accepted.

    ArduPilot silently ignores writes to parameter names it does not know, and
    the names churn between firmware releases (4.8 renamed ARSPD_FBW_MIN ->
    AIRSPEED_MIN, RLL2SRV_P -> RLL_RATE_P, LIM_PITCH_MAX -> PTCH_LIM_MAX_DEG,
    ...).  Without a read-back a stale name looks like it worked and the mission
    quietly flies on firmware defaults, which is exactly what happened here.
    """
    rejected, mismatched = [], []
    for name, val in params.items():
        m.mav.param_set_send(
            m.target_system, m.target_component,
            name, float(val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        # ArduPilot echoes a PARAM_VALUE for every accepted write.
        deadline, ack = time.time() + 1.0, None
        while time.time() < deadline:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.3)
            if msg and msg.param_id.strip("\x00") == name.decode():
                ack = msg
                break
        if ack is None:
            rejected.append(name.decode())
        elif abs(ack.param_value - float(val)) > max(0.01, abs(float(val)) * 0.001):
            mismatched.append((name.decode(), float(val), ack.param_value))

    if rejected:
        print(f"\n    !! {len(rejected)} parametro(s) RECHAZADOS (no existen en este firmware):")
        for n in rejected:
            print(f"       - {n}")
    if mismatched:
        print(f"\n    !! {len(mismatched)} parametro(s) con valor distinto al pedido:")
        for n, want, got in mismatched:
            print(f"       - {n}: pedido {want} -> quedo en {got}")
    if not rejected and not mismatched:
        print(f"    {len(params)} parametros aplicados y verificados.")


class FlightLog:
    """Telemetry shared by every phase of the mission, from arming to the end.

    feed() takes any MAVLink message; every ATTITUDE appends one CSV row with
    the latest value of everything else. Every wait in the mission goes through
    it, so the CSV covers takeoff and climb too -- before, the log only started
    after the climb, so the reports never showed the aircraft leaving the ground.
    """

    COLUMNS = ["time_s", "roll_deg", "pitch_deg", "airspeed", "altitude",
               "lat", "lon", "throttle_pct", "alt_agl", "thrust_n",
               "yaw_deg", "course_deg", "groundspeed"]

    def __init__(self, max_thrust=None):
        # thrust_n = max_thrust * (throttle/100)^2, the launcher's quadratic
        # curve. Written directly so the log does not depend on remembering
        # which prop_max_thrust was in force. Empty if the cap is unknown.
        self.max_thrust = max_thrust
        self.t0 = time.time()
        self.rows = []
        # altitude is VFR_HUD, i.e. MSL. agl is GLOBAL_POSITION_INT.relative_alt,
        # height above home, and home is where the aircraft armed, on the ground.
        # yaw (ATTITUDE) vs course (GLOBAL_POSITION_INT vx, vy): with no wind
        # in SITL their difference is the sideslip, measured instead of being
        # inferred from the orbit radius.
        self.cur = {"roll": 0, "pitch": 0, "spd": 0, "alt": 0, "lat": 0,
                    "lon": 0, "thr": 0, "agl": 0, "yaw": 0, "crs": 0, "gs": 0}
        self.max_r = self.max_p = 0.0

    def elapsed(self):
        return time.time() - self.t0

    def feed(self, msg):
        """Record one message. Returns its type, or None for a timeout."""
        if not msg:
            return None
        mt, c = msg.get_type(), self.cur
        if mt == "ATTITUDE":
            c["roll"], c["pitch"] = math.degrees(msg.roll), math.degrees(msg.pitch)
            c["yaw"] = math.degrees(msg.yaw) % 360
            self.max_r = max(self.max_r, abs(c["roll"]))
            self.max_p = max(self.max_p, abs(c["pitch"]))
            self.rows.append(dict(time=self.elapsed(), **c))
        elif mt == "VFR_HUD":
            c["spd"], c["alt"], c["thr"] = msg.airspeed, msg.alt, msg.throttle
        elif mt == "GLOBAL_POSITION_INT":
            c["lat"], c["lon"] = msg.lat / 1e7, msg.lon / 1e7
            c["agl"] = msg.relative_alt / 1000.0
            c["crs"] = math.degrees(math.atan2(msg.vy, msg.vx)) % 360
            c["gs"] = math.hypot(msg.vx, msg.vy) / 100.0
        return mt

    def pump(self, m, secs):
        """Keep logging for `secs` seconds. Use instead of time.sleep()."""
        end = time.time() + secs
        while time.time() < end:
            self.feed(m.recv_match(blocking=True,
                                   timeout=max(0.01, min(0.1, end - time.time()))))

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLUMNS)
            for r in self.rows:
                w.writerow([
                    f"{r['time']:.3f}", f"{r['roll']:.2f}", f"{r['pitch']:.2f}",
                    f"{r['spd']:.2f}",  f"{r['alt']:.2f}",
                    f"{r['lat']:.7f}",  f"{r['lon']:.7f}",
                    f"{r['thr']:.0f}",  f"{r['agl']:.2f}",
                    "" if self.max_thrust is None
                    else f"{self.max_thrust * (r['thr'] / 100) ** 2:.3f}",
                    f"{r['yaw']:.2f}", f"{r['crs']:.2f}", f"{r['gs']:.2f}",
                ])


def send_guided_wp(m, lat, lon, alt):
    m.mav.mission_item_send(
        m.target_system, m.target_component,
        0,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        2, 1,           # current=2 → guided target
        0, 50, 0, 0,    # hold, accept_r, pass, yaw
        lat, lon, alt,
    )


def send_guided_loiter(m, lat, lon, alt, radius):
    """NAV_LOITER_UNLIM around (lat, lon), NOT NAV_WAYPOINT. A plain waypoint
    only starts an implicit loiter once the aircraft arrives inside its accept
    radius — if the turn radius at the current airspeed is bigger than the
    requested WP_LOITER_RAD (as it was here: ~130m turn vs. a 60-100m orbit),
    the aircraft can never close the loop and just drifts outward forever
    instead of circling. This command tells the L1/loiter controller to orbit
    these exact coordinates immediately, independent of how it got there.
    """
    m.mav.mission_item_send(
        m.target_system, m.target_component,
        0,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
        2, 1,               # current=2 → guided target
        0, 0, radius, 0,    # empty, empty, radius (+CW/-CCW), yaw
        lat, lon, alt,
    )


def set_mode(m, name):
    if name not in m.mode_mapping():
        print(f"    !! mode {name} not available in this firmware")
        return False
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        m.mode_mapping()[name],
    )
    return True


def arm(m, log, timeout=15):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0,
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if log.feed(msg) != "HEARTBEAT" or msg.type == mavutil.mavlink.MAV_TYPE_GCS:
            continue
        if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("    ARMED!")
            return True
    print(f"    !! not armed after {timeout}s")
    return False


def takeoff_mode(m, log, alt, timeout=90):
    """ArduPlane TAKEOFF mode: ground roll, rotate at TKOFF_ROTATE_SPD, climb
    at TKOFF_LVL_PITCH with at most TKOFF_THR_MAX, level off at TKOFF_ALT.

    Replaces the old FBWA takeoff (4 s of 100 % throttle, then GUIDED). With
    T/W = 2.9 that reached 32.7 m/s, twice the cruise speed, and TECS turned
    the surplus into height: 48 deg of pitch and a climb to 65 m against a
    40 m target (auto_20260923_230914)."""
    print(f"[2] TAKEOFF mode, arming — climb to {alt:.0f} m …")
    set_mode(m, "TAKEOFF")
    log.pump(m, 0.5)
    if not arm(m, log):
        return False
    t0 = last = time.time()
    c = log.cur
    while time.time() - t0 < timeout:
        log.feed(m.recv_match(blocking=True, timeout=0.5))
        if time.time() - last >= 2:
            last = time.time()
            print(f"      AGL = {c['agl']:5.1f} m   V = {c['spd']:4.1f} m/s   "
                  f"Pitch = {c['pitch']:+5.1f}   Thr = {c['thr']:3.0f} %")
        # TAKEOFF hands over to normal flight 2 m below TKOFF_ALT and keeps
        # climbing to it. LOITER holds whatever height it is engaged at, so
        # wait for the last metres: at alt - 3 the orbit sat at 37 m, not 40
        # (auto_20260923_232856).
        if c["agl"] >= alt - 1:
            print(f"    Takeoff complete ({c['agl']:.0f} m, {c['spd']:.1f} m/s).")
            return True
    print(f"    !! {timeout}s without reaching {alt:.0f} m, continuing anyway.")
    return False


def fbwa_takeoff(m, log):
    """The original v1 takeoff, kept only for --replicate: FBWA, 4 s of full
    throttle by RC override, no speed limit."""
    print("[1] FBWA mode, arming …")
    set_mode(m, "FBWA")
    log.pump(m, 0.5)
    arm(m, log)

    TAKEOFF_SECS = 4
    print(f"[2] FBWA takeoff ({TAKEOFF_SECS}s, throttle 100 %) …")
    for _ in range(TAKEOFF_SECS * 10):
        m.mav.rc_channels_override_send(
            m.target_system, m.target_component,
            1500, 1500, 2000, 1500, 0, 0, 0, 0,
        )
        log.pump(m, 0.1)


def fly_to_wp(m, name, lat, lon, alt, log, timeout=45, accept=100):
    """Fly to a waypoint, logging every sample into the mission-wide log."""
    print(f"\n    >> {name}: ({lat:.6f}, {lon:.6f})")
    send_guided_wp(m, lat, lon, alt)
    log.pump(m, 0.3)

    t0 = time.time()
    c = log.cur
    while time.time() - t0 < timeout:
        if not log.feed(m.recv_match(blocking=True, timeout=0.5)):
            continue
        elapsed = time.time() - t0
        if c["lat"] and int(elapsed) % 2 == 0:
            d_wp   = dist_m(c["lat"], c["lon"], lat, lon)
            d_home = dist_m(c["lat"], c["lon"], HOME[0], HOME[1])
            inv    = " !! INVERTED" if abs(c["roll"]) > 120 else ""
            sys.stdout.write(
                f"\r       [{elapsed:4.0f}s] Dist={d_wp:5.0f}m "
                f"Roll={c['roll']:+6.1f} Pitch={c['pitch']:+6.1f} "
                f"V={c['spd']:4.0f}m/s Home={d_home:5.0f}m{inv}   ")
            sys.stdout.flush()
            if d_wp < accept:
                print(f"\n       >> Reached {name}!")
                return True

    if c["lat"]:
        print(f"\n       Timeout (dist={dist_m(c['lat'], c['lon'], lat, lon):.0f}m)")
    else:
        print("\n       Timeout (no GPS)")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    straight = "--straight" in sys.argv
    # --replicate reproduces the original v1 mission exactly as first flown.
    # That matters for more than the values:
    # back then most of these parameter names were the pre-4.8 spellings and
    # ArduPilot rejected them silently, so the aircraft actually flew on firmware
    # defaults for the PID gains, pitch limits and airspeed envelope.  Sending
    # the corrected names changes the flight even when the numbers are the same,
    # so replication means sending the original names and letting them bounce.
    replicate = "--replicate" in sys.argv
    # --loiter-now skips the waypoint legs entirely and engages LOITER mode right
    # after takeoff, orbiting wherever the aircraft happens to be.  The result
    # of interest is the loiter *manoeuvre*, not about orbiting a specific
    # coordinate, and the approach legs were what kept failing: the aircraft
    # never got closer than ~150 m to the commanded point, so the orbit started
    # 1.3 km away and never settled.  Trajectory analysis shows the aircraft does
    # fly real closed turns of 68-102 m radius at 24-28 m/s and -25 to -33 deg of
    # bank -- they just get interrupted before completing a revolution.
    loiter_now = "--loiter-now" in sys.argv
    # --thr-cap <pct> limits THR_MAX.  Binning |roll| by airspeed over all 20
    # logs shows a clear stability envelope: between 10 and 20 m/s the median
    # bank is 10-15 deg and under 1 % of samples exceed 45 deg, but above 30 m/s
    # the p95 bank jumps to 82 deg and 20 % of samples pass 45 deg (i.e. the roll
    # axis diverges).  The only log that never exceeded 45 deg, auto_20260817_180446,
    # is also the only one that averaged 15.8 m/s instead of 27-30.
    #
    # The aircraft has no way to stay inside that envelope at THR_MAX=100: the
    # sim's 15 N of static thrust on a 6.01 N airframe is T/W = 2.5, against the
    # 0.5-0.8 of a real foamboard trainer.  Drag at the 15 m/s design cruise is
    # only 2.44 N (16 % throttle), so any climb demand runs the speed away.
    # Capping at 30 % gives 4.5 N -> T/W = 0.75, a level-flight ceiling of
    # ~20 m/s, and still 2.1 N of excess thrust at cruise (~5 m/s of climb).
    thr_cap = None
    if "--thr-cap" in sys.argv:
        thr_cap = float(sys.argv[sys.argv.index("--thr-cap") + 1])
    # --duration <s> lets the mission end on its own instead of needing Ctrl+C,
    # so runs can be scripted back-to-back.
    duration = None
    if "--duration" in sys.argv:
        duration = float(sys.argv[sys.argv.index("--duration") + 1])
    # --ccw and --radius <m> override LOITER_DIR / LOITER_RADIUS, so the
    # counter-clockwise orbit (auto_20260923_232856: CCW, 150 m) can be re-flown
    # without editing this file.
    global LOITER_DIR, LOITER_RADIUS
    if "--ccw" in sys.argv:
        LOITER_DIR = -1
    if "--radius" in sys.argv:
        LOITER_RADIUS = float(sys.argv[sys.argv.index("--radius") + 1])
    print("=" * 60)
    print("  AUTONOMOUS FLIGHT — YOY Trainer v2")
    if loiter_now:
        print("  LOITER-NOW — orbits wherever the aircraft is after takeoff")
        print("=" * 60)
    elif straight:
        print("  STRAIGHT-FLIGHT DIAGNOSTIC — no turns, isolates TECS/pitch")
        print("=" * 60)
        print(f"  Target: {STRAIGHT_WP}  (~45 km east, never reached)")
    else:
        print("  Mission Planner waypoints")
        print("=" * 60)
        print(f"  WP0:    {WP0}  (east, straight)")
        print(f"  WP1:    {WP1}  (south, turn)")
        print(f"  LOITER: {LOITER}  (circles)")
    print("=" * 60)

    m = connect()

    # Mission-wide log: starts before arming so the CSV shows the takeoff roll
    # and the climb, and the ground level is in the data, not guessed.
    ts       = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"auto_{ts}.csv")
    # Written by the launcher at start-up: the plant actually being simulated.
    sim_config_path = os.path.join(LOG_DIR, "sim_config.json")
    try:
        with open(sim_config_path) as f:
            sim_config = json.load(f)
        print(f"  Plant: prop_max_thrust = {sim_config['prop_max_thrust_n']} N "
              f"(sim_config.json, {sim_config['written']})")
    except (OSError, ValueError, KeyError):
        sim_config = None
        print("  WARNING: flight_logs/sim_config.json missing -- thrust_n will "
              "be empty. Relaunch the simulator with the current launcher.")
    log      = FlightLog(sim_config["prop_max_thrust_n"] if sim_config else None)
    for sid, rate in [
        (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,   20),
        (mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,   10),
        (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10),
    ]:
        m.mav.request_data_stream_send(
            m.target_system, m.target_component, sid, rate, 1)

    # Phase 0 — disable arming checks for SITL
    # ARMING_CHECK (bitmask of checks to RUN) was replaced by ARMING_SKIPCHK
    # (bitmask of checks to SKIP) in 4.7 — 0 now means "skip nothing", the
    # opposite of the old default. 2097151 = bits 1..20 set = every check
    # ArduPilot 4.8 knows about (see AP_Arming::Check in AP_Arming.h).
    print("[0] Configuring …")
    set_params(m, {b"ARMING_CHECK": 0.0} if replicate
                   else {b"ARMING_SKIPCHK": 2097151.0})
    time.sleep(0.5)

    if replicate:
        # Original order: FBWA takeoff first, parameters afterwards.
        fbwa_takeoff(m, log)
        print("[3] Switching to GUIDED + flight parameters …")
    else:
        # Parameters BEFORE arming, so the takeoff already flies with the
        # final limits and gains.
        print("[1] Flight + takeoff parameters …")
    set_params(m, {
        # NOTE: names below are for ArduPlane 4.8.  Several were renamed after
        # 4.3 and the old spellings are silently ignored — set_params() now
        # verifies each write, so a bad name shows up as a "RECHAZADO" line.
        #   LIM_ROLL_CD    -> ROLL_LIMIT_DEG    (degrees, not centidegrees)
        #   LIM_PITCH_MAX  -> PTCH_LIM_MAX_DEG  (degrees)
        #   TRIM_PITCH_CD  -> PTCH_TRIM_DEG     (degrees)
        #   ARSPD_FBW_MIN  -> AIRSPEED_MIN
        #   ARSPD_FBW_MAX  -> AIRSPEED_MAX
        #   TRIM_ARSPD_CM  -> AIRSPEED_CRUISE   (m/s, not cm/s)
        #   RLL2SRV_P/I/D  -> RLL_RATE_P/I/D
        #   PTCH2SRV_P/I/D -> PTCH_RATE_P/I/D
        b"ROLL_LIMIT_DEG":    35.0,
        b"PTCH_LIM_MAX_DEG":  20.0,
        b"PTCH_LIM_MIN_DEG": -15.0,
        b"PTCH_TRIM_DEG":      0.0,
        b"NAVL1_PERIOD":      25.0,
        b"WP_RADIUS":         80.0,
        b"WP_LOITER_RAD":     LOITER_RADIUS * LOITER_DIR,
        b"PTCH2SRV_RLL":       1.0,
        b"AIRSPEED_MIN":       9.0,
        b"AIRSPEED_MAX":      20.0,
        b"AIRSPEED_CRUISE":   15.0,
        # Cruise throttle at 15 m/s for v2 with 100 % = 8.585 N (quadratic):
        # 2.91 N of drag (1.41 aero from the CFD polar + 1.50 from Pegasus's
        # default LinearDrag) -> sqrt(2.91/8.585) = 58 %. Drops to ~41 % if the
        # LinearDrag is removed. The v1 value, 18 %, sat on a thrust curve
        # that saturated at 15 %, i.e. it was a full-power trim.
        b"TRIM_THROTTLE":     58.0,
        b"THR_MAX":          100.0 if thr_cap is None else thr_cap,
        b"THR_MIN":            0.0,
        # Rate loops: ArduPlane 4.8 defaults (AP_RollController.cpp /
        # AP_PitchController.cpp). The v1 values, P = 0.4 and D = 0.04, were
        # 5x (roll) and 10x (pitch) the default P and rang at 5 Hz in bursts
        # every 2.6 s (auto_20260923_230914). Written explicitly, not left
        # out: SITL keeps set parameters in its EEPROM between runs. These are
        # the baseline for AUTOTUNE, not a tune.
        b"RLL_RATE_P":         0.08,
        b"RLL_RATE_I":         0.15,
        b"RLL_RATE_D":         0.0,
        b"RLL_RATE_FF":        0.345,
        b"PTCH_RATE_P":        0.04,
        b"PTCH_RATE_I":        0.15,
        b"PTCH_RATE_D":        0.0,
        b"PTCH_RATE_FF":       0.345,
        b"TECS_PTCH_DAMP":     0.6,
        b"TECS_TIME_CONST":    7.0,
        # TAKEOFF mode. 80 % throttle is 5.5 N with the quadratic thrust
        # curve, T/W = 1.07: a real trainer climb-out, where 100 % (8.585 N)
        # is T/W = 1.67. (60 % would now be only 3.1 N, T/W = 0.60.) Rotate at 9 m/s = 1.15 x the 7.84 m/s stall with gear.
        b"TKOFF_ALT":          CLIMB_ALT,
        b"TKOFF_THR_MAX":     80.0,
        # Throttle ramp during the takeoff roll, %/s. The default (0 -> the
        # 100 %/s of THR_SLEWRATE) reached 80 % in under a second at ~1 m/s:
        # the nose rose to +23 deg, the aircraft hopped 0.24 m off the contact
        # and nosed over to -38..-53 deg in 3 of the 4 runs of 2026-09-25.
        # 20 %/s = 0 -> 80 % in 4 s, the lowest value ArduPilot recommends.
        b"TKOFF_THR_SLEW":    20.0,
        b"TKOFF_ROTATE_SPD":   9.0,
        b"TKOFF_LVL_PITCH":   12.0,
    } if not replicate else {
        # Verbatim from the original v1 mission. Most of these names no longer exist in 4.8 and
        # will be reported as rejected — that is the point: it is what the
        # aircraft actually flew with.
        b"ROLL_LIMIT_DEG":  35.0,
        b"LIM_ROLL_CD":     3500.0,
        b"LIM_PITCH_MAX":   2000.0,
        b"LIM_PITCH_MIN":  -1500.0,
        b"TRIM_PITCH_CD":   0.0,
        b"NAVL1_PERIOD":    25.0,
        b"WP_RADIUS":       80.0,
        b"WP_LOITER_RAD":   100.0,
        b"PTCH2SRV_RLL":    1.0,
        b"ARSPD_FBW_MIN":   12.0,
        b"ARSPD_FBW_MAX":   30.0,
        b"TRIM_THROTTLE":   65.0,
        b"THR_MAX":         100.0,
        b"THR_MIN":         0.0,
        b"PTCH2SRV_P":      0.8,
        b"PTCH2SRV_I":      0.05,
        b"PTCH2SRV_D":      0.02,
        b"RLL2SRV_P":       0.8,
        b"RLL2SRV_I":       0.1,
        b"RLL2SRV_D":       0.02,
    })
    log.pump(m, 0.3)

    if replicate:
        set_mode(m, "GUIDED")
        log.pump(m, 0.5)
        m.mav.rc_channels_override_send(
            m.target_system, m.target_component, 0, 0, 0, 0, 0, 0, 0, 0,
        )
        log.pump(m, 0.5)
    else:
        takeoff_mode(m, log, CLIMB_ALT)
        if not loiter_now:
            set_mode(m, "GUIDED")
            log.pump(m, 0.5)

    if loiter_now:
        # Climb BEFORE engaging LOITER.  The first --loiter-now run
        # (auto_20260817_180446) switched to LOITER straight after the 4 s
        # takeoff, and LOITER holds whatever altitude it inherits -- which was
        # zero.  The aircraft orbited for 240 s between -3 and +2 m AGL.  It was
        # the most stable log on record (15.8 m/s, never past 41 deg of bank,
        # turn radii matching V^2/(g*tan(phi)) to within 18 %), but only because
        # never climbing meant TECS never opened the throttle, and the aircraft
        # is only stable below ~20 m/s.  A loiter at zero altitude is not a
        # loiter, so climb first, then orbit at height.  TAKEOFF mode already
        # did; only --replicate (FBWA takeoff) still needs the GUIDED climb.
        climb_target = HOME[0] - 0.008, HOME[1]      # ~900 m south, gentle leg
        if replicate:
            print(f"\n[4] Climbing to {CLIMB_ALT:.0f} m before engaging LOITER …")
            send_guided_wp(m, climb_target[0], climb_target[1], CLIMB_ALT)
            # Height above home (relative_alt), not above the first VFR_HUD sample:
            # that one arrives after the 4 s takeoff, already several metres up,
            # so "34 m above it" was never reached and every run burned the whole
            # 120 s timeout circling at 40 m before LOITER engaged.
            t_climb = last = time.time()
            while time.time() - t_climb < 120:
                log.feed(m.recv_match(blocking=True, timeout=0.5))
                agl = log.cur["agl"]
                if time.time() - last >= 3:
                    last = time.time()
                    print(f"      AGL = {agl:5.0f} m   V = {log.cur['spd']:4.1f} m/s   "
                          f"(objetivo {CLIMB_ALT:.0f} m)")
                if agl >= CLIMB_ALT * 0.85:
                    print(f"    Altura alcanzada ({agl:.0f} m) — enganchando LOITER.")
                    break
            else:
                print("    !! 120 s sin alcanzar altura, enganchando LOITER igual.")
        set_mode(m, "LOITER")
        print("\n[5] Monitoring the orbit.  Press Ctrl+C to stop.\n")
        dist_ref = LOITER
    elif straight:
        # Diagnostic mode: one distant waypoint, never turning, to isolate
        # whether the altitude/pitch oscillation is a turn-induced (lateral)
        # problem or a longitudinal (TECS/pitch) problem that happens anyway.
        print("\n[4] Flying straight — no waypoints, no loiter.")
        send_guided_wp(m, STRAIGHT_WP[0], STRAIGHT_WP[1], FLIGHT_ALT)
        print("\n[5] Monitoring straight flight.  Press Ctrl+C to stop.\n")
        dist_ref = HOME
    else:
        # Phase 4 — fly waypoints
        print("\n[4] Executing waypoint trajectory:")
        print("    HOME → WP0 (east) → WP1 (south) → LOITER (circles)")
        fly_to_wp(m, "WP0 East",       WP0[0],    WP0[1],    FLIGHT_ALT, log,
                  timeout=30)
        fly_to_wp(m, "WP1 South",      WP1[0],    WP1[1],    FLIGHT_ALT, log,
                  timeout=40)
        fly_to_wp(m, "LOITER Circles", LOITER[0], LOITER[1], FLIGHT_ALT, log,
                  timeout=50)

        # LOITER mode circles around wherever the aircraft IS when the mode
        # engages, not around LOITER's coordinates — switching right after a
        # timed-out approach leg (as before) orbited 1600m away from the
        # target. Keep commanding the GUIDED target and wait until we're
        # actually close (within 1.5x the loiter radius) before switching
        # modes, instead of relying on fly_to_wp's fixed 50s timeout above.
        print("\n    Waiting to get close before engaging LOITER mode …")
        wait_deadline = time.time() + (0 if replicate else 90)
        close_enough = LOITER_RADIUS * 1.5
        wait_deadline = time.time() + 90
        while time.time() < wait_deadline:
            if log.feed(m.recv_match(blocking=True, timeout=0.5)) != "GLOBAL_POSITION_INT":
                continue
            d = dist_m(log.cur["lat"], log.cur["lon"], LOITER[0], LOITER[1])
            if int(time.time()) % 3 == 0:
                print(f"      dist to LOITER = {d:5.0f} m  (need < {close_enough:.0f} m)")
            if d < close_enough:
                print(f"    Close enough ({d:.0f} m) — engaging LOITER.")
                break
        else:
            print(f"    !! 90s wait expired, engaging LOITER wherever we are now.")

        # Phase 5 — switch to the native LOITER flight mode. GUIDED + a
        # NAV_LOITER_UNLIM mission item (tried previously) did not produce a
        # closed orbit — the aircraft flew one big, slow arc instead of
        # circling, so whatever GUIDED does with that command type is not an
        # orbit hold. An actual flight-mode change is unambiguous:
        # ArduPilot's LOITER mode always circles at WP_LOITER_RAD around
        # wherever the aircraft is when the mode engages, independent of
        # mission-item semantics.
        if replicate:
            print(f"\n\n[5] Circling over LOITER (GUIDED waypoint, as in the original mission).\n")
            send_guided_wp(m, LOITER[0], LOITER[1], FLIGHT_ALT)
            dist_ref = LOITER
            replicate_done = True
        else:
            replicate_done = False
        print(f"\n\n[5] Switching to LOITER mode.  Press Ctrl+C to stop.\n")
        if replicate_done:
            pass
        elif "LOITER" in m.mode_mapping():
            m.mav.set_mode_send(
                m.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                m.mode_mapping()["LOITER"],
            )
        else:
            print("    !! 'LOITER' not in mode_mapping() — falling back to GUIDED loiter command.")
            send_guided_loiter(m, LOITER[0], LOITER[1], FLIGHT_ALT, LOITER_RADIUS)
        dist_ref = LOITER

    last_print = 0
    cur = log.cur

    try:
        while True:
            if duration is not None and log.elapsed() > duration:
                print(f"\n  -- {duration:.0f}s reached, ending run --")
                break
            if log.feed(m.recv_match(blocking=True, timeout=1)) == "ATTITUDE":
                t = log.elapsed()
                if t - last_print >= 2:
                    last_print = t
                    d = (dist_m(cur["lat"], cur["lon"], dist_ref[0], dist_ref[1])
                         if cur["lat"] else 9999)
                    label = "HOME" if straight else "LOIT"
                    inv = " INVERTED!" if abs(cur["roll"]) > 120 else ""
                    bar = "=" * min(int(abs(cur["roll"]) / 5), 20)
                    print(f"  [{t:5.0f}s] Roll={cur['roll']:+7.1f} "
                          f"[{bar:20s}] Pitch={cur['pitch']:+6.1f} "
                          f"V={cur['spd']:4.0f}m/s AGL={cur['agl']:5.0f}m "
                          f"Thr={cur['thr']:3.0f}% "
                          f"{label}={d:5.0f}m{inv}")

    except KeyboardInterrupt:
        pass

    print(f"\n\nSaving log ({len(log.rows)} samples) → {os.path.basename(csv_path)}")
    log.save(csv_path)
    # Sidecar with everything needed to interpret the CSV without the code as
    # it was on the day: the plant and the mission geometry.
    with open(csv_path[:-4] + ".json", "w") as f:
        json.dump({"sim_config": sim_config, "argv": sys.argv[1:],
                   "loiter_dir": LOITER_DIR, "loiter_radius_m": LOITER_RADIUS,
                   "climb_alt_m": CLIMB_ALT}, f, indent=2)
    # Pegasus writes the TRUE state (sideslip, alpha, forces, attitude) to
    # forces_log.csv in the directory isaac_run was started from, and
    # overwrites it on every launch. ArduPilot's yaw is an EKF estimate that
    # wanders by ~1 deg, too much to measure a sub-degree sideslip, so keep
    # the ground truth next to the CSV.
    for cand in (os.path.join(SCRIPT_DIR, os.pardir, os.pardir, os.pardir, "forces_log.csv"),
                 os.path.join(os.getcwd(), "forces_log.csv")):
        if os.path.exists(cand) and time.time() - os.path.getmtime(cand) < 60:
            with open(cand, "rb") as src, gzip.open(csv_path[:-4] + "_forces.csv.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  Pegasus ground truth → {os.path.basename(csv_path[:-4])}_forces.csv.gz")
            break
    else:
        print("  (forces_log.csv not found or stale -- no ground-truth copy)")

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Max roll:   {log.max_r:.1f}°")
    print(f"  Max pitch:  {log.max_p:.1f}°")
    print(f"  Inverted:   {'YES' if log.max_r > 170 else 'NO'}")
    print(f"  Log:        {csv_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
