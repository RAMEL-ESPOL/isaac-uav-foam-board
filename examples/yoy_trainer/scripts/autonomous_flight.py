#!/usr/bin/env python
"""
| File: autonomous_flight.py
| Author: Steven Martinez Jara
| License: BSD-3-Clause
| Description: Autonomous waypoint mission + LOITER monitor for the YOY Trainer.
|              Connects to ArduPilot SITL via MAVLink, arms in FBWA, takes off,
|              flies WP0 → WP1 → LOITER, and logs telemetry to CSV.
|
| Prerequisites (run both from the repo root):
|   Terminal 1:  isaac_run examples/yoy_trainer/13_yoy_trainer_fixedwing.py --mode autonomous
|   Terminal 2:  python3   examples/yoy_trainer/scripts/autonomous_flight.py
"""

import sys
import time
import csv
import os
import math

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
LOITER_DIR    = -1      # -1 = counter-clockwise (left turns), +1 = clockwise.
                         # Trajectory analysis of all 9 logs shows LEFT banks
                         # produce the correct turn rate (actual/theoretical
                         # g*tan(phi)/V ratio ~1.1-1.4) while RIGHT banks
                         # produce almost none (ratio ~0.0 to -0.5), i.e. the
                         # sim's right-hand turn dynamics are broken. Until
                         # that is fixed, orbit counter-clockwise so the
                         # aircraft only ever has to turn the direction that
                         # actually works.
LOITER_RADIUS = 150.0   # m  — AIRSPEED_CRUISE=15 is verified-applied but the
                         # aircraft is still flying flat at ~29-30 m/s the
                         # entire mission (TECS not converging; root cause not
                         # yet found). Min turn radius at that speed and
                         # ROLL_LIMIT_DEG=35 is ~126 m, so 150 m is achievable
                         # *today* without depending on the speed bug getting
                         # fixed. If TECS gets sorted out later this can drop
                         # back toward 60 m (min radius at 15 m/s is ~33 m).
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


def drain(m, secs=0.5):
    t0 = time.time()
    while time.time() - t0 < secs:
        m.recv_match(blocking=True, timeout=0.1)


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


def fly_to_wp(m, name, lat, lon, alt, timeout=45, accept=100,
              rows=None, cur=None, t0_mission=None, stats=None):
    """Fly to a waypoint. If rows/cur/t0_mission/stats are given, every ATTITUDE
    sample is also appended to the shared mission log (same format used by the
    LOITER phase), so the CSV covers the whole flight, not just the loiter."""
    print(f"\n    >> {name}: ({lat:.6f}, {lon:.6f})")
    send_guided_wp(m, lat, lon, alt)
    time.sleep(0.3)

    t0 = time.time()
    cur_lat = cur_lon = cur_roll = cur_pitch = cur_spd = 0

    while time.time() - t0 < timeout:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if not msg:
            continue
        mt = msg.get_type()
        if mt == "GLOBAL_POSITION_INT":
            cur_lat, cur_lon = msg.lat / 1e7, msg.lon / 1e7
            if cur is not None:
                cur["lat"], cur["lon"] = cur_lat, cur_lon
        elif mt == "ATTITUDE":
            cur_roll  = math.degrees(msg.roll)
            cur_pitch = math.degrees(msg.pitch)
            if cur is not None:
                cur["roll"], cur["pitch"] = cur_roll, cur_pitch
                if stats is not None:
                    stats["max_r"] = max(stats["max_r"], abs(cur_roll))
                    stats["max_p"] = max(stats["max_p"], abs(cur_pitch))
                if rows is not None and t0_mission is not None:
                    rows.append(dict(time=time.time() - t0_mission, **cur))
        elif mt == "VFR_HUD":
            cur_spd = msg.airspeed
            if cur is not None:
                cur["spd"] = cur_spd
                cur["alt"] = msg.alt

        elapsed = time.time() - t0
        if cur_lat and int(elapsed) % 2 == 0:
            d_wp   = dist_m(cur_lat, cur_lon, lat, lon)
            d_home = dist_m(cur_lat, cur_lon, HOME[0], HOME[1])
            inv    = " !! INVERTED" if abs(cur_roll) > 120 else ""
            sys.stdout.write(
                f"\r       [{elapsed:4.0f}s] Dist={d_wp:5.0f}m "
                f"Roll={cur_roll:+6.1f} Pitch={cur_pitch:+6.1f} "
                f"V={cur_spd:4.0f}m/s Home={d_home:5.0f}m{inv}   ")
            sys.stdout.flush()
            if d_wp < accept:
                print(f"\n       >> Reached {name}!")
                return True

    if cur_lat:
        print(f"\n       Timeout (dist={dist_m(cur_lat, cur_lon, lat, lon):.0f}m)")
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
    print("=" * 60)
    print("  AUTONOMOUS FLIGHT — YOY Trainer")
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

    # Phase 0 — disable arming checks for SITL
    # ARMING_CHECK (bitmask of checks to RUN) was replaced by ARMING_SKIPCHK
    # (bitmask of checks to SKIP) in 4.7 — 0 now means "skip nothing", the
    # opposite of the old default. 2097151 = bits 1..20 set = every check
    # ArduPilot 4.8 knows about (see AP_Arming::Check in AP_Arming.h).
    print("[0] Configuring …")
    set_params(m, {b"ARMING_CHECK": 0.0} if replicate
                   else {b"ARMING_SKIPCHK": 2097151.0})
    time.sleep(0.5)

    # Phase 1 — arm in FBWA
    print("[1] FBWA mode, arming …")
    if "FBWA" in m.mode_mapping():
        m.mav.set_mode_send(
            m.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            m.mode_mapping()["FBWA"],
        )
    time.sleep(0.5)
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0,
    )
    t0 = time.time()
    while time.time() - t0 < 15:
        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if msg and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
            if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                print("    ARMED!")
                break

    # Phase 2 — FBWA takeoff
    TAKEOFF_SECS = 4
    print(f"[2] FBWA takeoff ({TAKEOFF_SECS}s, throttle 100 %) …")
    for _ in range(TAKEOFF_SECS * 10):
        m.mav.rc_channels_override_send(
            m.target_system, m.target_component,
            1500, 1500, 2000, 1500, 0, 0, 0, 0,
        )
        time.sleep(0.1)

    # Phase 3 — switch to GUIDED + flight parameters
    print("[3] Switching to GUIDED + flight parameters …")
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
        b"TRIM_THROTTLE":     18.0,
        b"THR_MAX":          100.0 if thr_cap is None else thr_cap,
        b"THR_MIN":            0.0,
        b"PTCH_RATE_P":        0.4,
        b"PTCH_RATE_I":        0.05,
        b"PTCH_RATE_D":        0.04,
        b"RLL_RATE_P":         0.4,
        b"RLL_RATE_I":         0.1,
        b"RLL_RATE_D":         0.04,
        b"TECS_PTCH_DAMP":     0.6,
        b"TECS_TIME_CONST":    7.0,
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
    time.sleep(0.3)

    if "GUIDED" in m.mode_mapping():
        m.mav.set_mode_send(
            m.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            m.mode_mapping()["GUIDED"],
        )
    time.sleep(0.5)

    m.mav.rc_channels_override_send(
        m.target_system, m.target_component, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    drain(m, 0.5)

    # Mission-wide log — starts now, covers waypoint transit AND loiter, not
    # just the loiter phase (previously time_s reset to 0 when loiter began,
    # so the CSV silently dropped the WP0/WP1 transit legs).
    ts       = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"auto_{ts}.csv")
    rows     = []
    cur      = {"roll": 0, "pitch": 0, "spd": 0, "alt": 0, "lat": 0, "lon": 0}
    stats    = {"max_r": 0, "max_p": 0}
    t0       = time.time()

    for sid, rate in [
        (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,   20),
        (mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,   10),
        (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10),
    ]:
        m.mav.request_data_stream_send(
            m.target_system, m.target_component, sid, rate, 1)

    if loiter_now:
        # Climb BEFORE engaging LOITER.  The first --loiter-now run
        # (auto_20260817_180446) switched to LOITER straight after the 4 s
        # takeoff, and LOITER holds whatever altitude it inherits -- which was
        # zero.  The aircraft orbited for 240 s between -3 and +2 m AGL.  It was
        # the most stable log on record (15.8 m/s, never past 41 deg of bank,
        # turn radii matching V^2/(g*tan(phi)) to within 18 %), but only because
        # never climbing meant TECS never opened the throttle, and the aircraft
        # is only stable below ~20 m/s.  A loiter at zero altitude is not a
        # loiter, so climb first, then orbit at height.
        climb_target = HOME[0] - 0.008, HOME[1]      # ~900 m south, gentle leg
        print(f"\n[4] Climbing to {CLIMB_ALT:.0f} m before engaging LOITER …")
        send_guided_wp(m, climb_target[0], climb_target[1], CLIMB_ALT)
        t_climb = time.time()
        alt0 = None
        while time.time() - t_climb < 120:
            msg = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
            if not msg:
                continue
            if alt0 is None:
                alt0 = msg.alt
            agl = msg.alt - alt0
            if int(time.time() - t_climb) % 3 == 0:
                print(f"      AGL = {agl:5.0f} m   V = {msg.airspeed:4.1f} m/s   "
                      f"(objetivo {CLIMB_ALT:.0f} m)")
            if agl >= CLIMB_ALT * 0.85:
                print(f"    Altura alcanzada ({agl:.0f} m) — enganchando LOITER.")
                break
        else:
            print("    !! 120 s sin alcanzar altura, enganchando LOITER igual.")
        if "LOITER" in m.mode_mapping():
            m.mav.set_mode_send(
                m.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                m.mode_mapping()["LOITER"],
            )
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
        fly_to_wp(m, "WP0 East",       WP0[0],    WP0[1],    FLIGHT_ALT, timeout=30,
                  rows=rows, cur=cur, t0_mission=t0, stats=stats)
        fly_to_wp(m, "WP1 South",      WP1[0],    WP1[1],    FLIGHT_ALT, timeout=40,
                  rows=rows, cur=cur, t0_mission=t0, stats=stats)
        fly_to_wp(m, "LOITER Circles", LOITER[0], LOITER[1], FLIGHT_ALT, timeout=50,
                  rows=rows, cur=cur, t0_mission=t0, stats=stats)

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
            msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
            if not msg:
                continue
            d = dist_m(msg.lat / 1e7, msg.lon / 1e7, LOITER[0], LOITER[1])
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
    max_r, max_p = stats["max_r"], stats["max_p"]

    try:
        while True:
            if duration is not None and (time.time() - t0) > duration:
                print(f"\n  -- {duration:.0f}s reached, ending run --")
                break
            msg = m.recv_match(blocking=True, timeout=1)
            if not msg:
                continue
            mt = msg.get_type()

            if mt == "ATTITUDE":
                cur["roll"]  = math.degrees(msg.roll)
                cur["pitch"] = math.degrees(msg.pitch)
                max_r = max(max_r, abs(cur["roll"]))
                max_p = max(max_p, abs(cur["pitch"]))
                t = time.time() - t0
                rows.append(dict(time=t, **cur))
                if t - last_print >= 2:
                    last_print = t
                    d = (dist_m(cur["lat"], cur["lon"], dist_ref[0], dist_ref[1])
                         if cur["lat"] else 9999)
                    label = "HOME" if straight else "LOIT"
                    inv = " INVERTED!" if abs(cur["roll"]) > 120 else ""
                    bar = "=" * min(int(abs(cur["roll"]) / 5), 20)
                    print(f"  [{t:5.0f}s] Roll={cur['roll']:+7.1f} "
                          f"[{bar:20s}] Pitch={cur['pitch']:+6.1f} "
                          f"V={cur['spd']:4.0f}m/s Alt={cur['alt']:5.0f}m "
                          f"{label}={d:5.0f}m{inv}")

            elif mt == "VFR_HUD":
                cur["spd"] = msg.airspeed
                cur["alt"] = msg.alt

            elif mt == "GLOBAL_POSITION_INT":
                cur["lat"] = msg.lat / 1e7
                cur["lon"] = msg.lon / 1e7

    except KeyboardInterrupt:
        pass

    # Save CSV log
    print(f"\n\nSaving log ({len(rows)} samples) → {os.path.basename(csv_path)}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "roll_deg", "pitch_deg", "airspeed",
                     "altitude", "lat", "lon"])
        for r in rows:
            w.writerow([
                f"{r['time']:.3f}", f"{r['roll']:.2f}", f"{r['pitch']:.2f}",
                f"{r['spd']:.2f}",  f"{r['alt']:.2f}",
                f"{r['lat']:.7f}",  f"{r['lon']:.7f}",
            ])

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Max roll:   {max_r:.1f}°")
    print(f"  Max pitch:  {max_p:.1f}°")
    print(f"  Inverted:   {'YES' if max_r > 170 else 'NO'}")
    print(f"  Log:        {csv_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
