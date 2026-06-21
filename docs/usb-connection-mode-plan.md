# Plan: Choose USB connection type (fast USB3 vs forced USB2) live from the GUI

## Context

Today the OAK link is handled fully automatically: on every connect,
`camera.py:_tune_for_usb()` (camera.py:271) calls `device.getUsbSpeed()` and
trims the pipeline when it sees a USB2 (HighSpeed) link — capping FPS to 10 and
disabling extended disparity so the three streams fit the ~480 Mbps bus. The
USB3/USB2 handshake is flaky on marginal cables/ports, and there is no way for
the operator to override the guess.

The goal is to **choose the connection type at runtime from the on-screen GUI**,
and the choice should do **both**: force the actual USB link speed at
device-open *and* set matching pipeline tuning. Three modes:

- **AUTO** — current behavior (negotiate link, detect speed, adapt pipeline).
- **FAST** — allow USB3, force the full pipeline (full FPS + extended disparity).
- **USB2 (compat)** — lock the link to USB2 via `dai.Device(dai.UsbSpeed.HIGH)`
  (often fixes the intermittent USB3 handshake on bad cables) + reduced pipeline.

The reconnect needed for a live switch already exists: the model-switch command
returns `True` from `_handle_command`, which `_drain_commands` (camera.py:488)
uses to break `_loop`, closing the device `with`-block in `run()` (camera.py:390)
so the outer loop reconnects — with no error frame. We reuse this path exactly.

## Verified API (DepthAI v3)

`dai.Device(dai.UsbSpeed.HIGH)` constructs a device with the link capped to USB2.
A bare `dai.Device()` negotiates the best available speed (used for AUTO/FAST).
There is no way to *force* a USB3 link if the hardware won't negotiate it, so
FAST simply allows negotiation and forces the full pipeline regardless of the
detected speed (operator's explicit tradeoff: may stall if the link is really
USB2). Confirmed against the v3 device docs.

## Changes

### `config.py`
- Add `"usb_mode": "auto"` to `DEFAULTS` (config.py:13) with a short comment.
  Valid values: `auto` / `fast` / `compat`. This is only the **initial** value;
  the live control is the GUI toggle.

### `camera.py`
1. **`__init__`** (near camera.py:119, beside `_build_fps`): add
   `self._usb_mode = self.cfg.get("usb_mode", "auto")` and
   `self._desired_usb_mode = self._usb_mode` (mirrors the `_model` /
   `_desired_model` pairing).
2. **New `_open_device(self, dai)` helper**: returns
   `dai.Device(dai.UsbSpeed.HIGH)` when `self._usb_mode == "compat"`, else
   `dai.Device()`. Docstring noting USB2 lock helps marginal links.
3. **`run()`** (camera.py:384–392): under the existing lock, also set
   `self._usb_mode = self._desired_usb_mode`; replace
   `with dai.Device() as device:` with `with self._open_device(dai) as device:`.
4. **`_tune_for_usb()`** (camera.py:271): after the default
   `_build_fps/_extended_disparity` reset, branch on mode before the
   `getUsbSpeed()` probe:
   - `fast` → keep full settings, log `USB mode: FAST (forced)`, return.
   - `compat` → `_build_fps = min(self.fps, 10)`, `_extended_disparity = False`,
     log `USB mode: USB2 (forced)`, return.
   - `auto` → existing detect-and-adapt logic unchanged.
5. **`_handle_command()`** (near camera.py:264, beside `switch`): add
   `elif action == "usb_mode": self._desired_usb_mode = cmd[1]; return True`.
6. **`get_status()`** (camera.py:134): add `"usb_mode": self._usb_mode` so the UI
   can label the button.

### `app.py`
1. **Toolbar** (`specs`, app.py:74): add a 7th button
   `("usb", "USB: AUTO", "#3a3a3a", self.cycle_usb)`. Bar is 800px → ~114px per
   button; labels stay short (`USB: AUTO/FAST/USB2`).
2. **New `cycle_usb()`** + `_USB_MODES = ["auto", "fast", "compat"]`: read current
   mode from `self.worker.get_status()`, send the next one via
   `self.worker.send("usb_mode", nxt)`.
3. **`_refresh_buttons()`** (app.py:141): add `st["usb_mode"]` to the `state`
   tuple so the label refreshes, and set the button text from a
   `{auto:"USB: AUTO", fast:"USB: FAST", compat:"USB: USB2"}` map.

## Notes / tradeoffs
- Each tap triggers a brief reconnect (MODEL button shows `LOADING...`, then a
  `Camera ready` toast) — same UX as switching models.
- Mock mode (`--mock`) has no real device; the toggle just cycles the label and
  is otherwise a no-op there.
- FAST on a link that only negotiates USB2 may stall; `_loop`'s stale-frame
  watchdog (camera.py:429) will surface it and reconnect. This is the operator's
  explicit choice and will be noted in the button behavior.

## Verification
1. `python app.py --mock` — confirm the USB button appears, cycles
   AUTO→FAST→USB2→AUTO on tap, and the app doesn't crash (no hardware effect).
2. On the Pi with the OAK-D, `python app.py`:
   - Tap to **USB2**: console shows the device opening at HighSpeed and
     `USB mode: USB2 (forced) -> reduced pipeline @ 10 fps`; reconnect succeeds.
   - Tap to **FAST**: `USB mode: FAST (forced) -> full pipeline @ 20 fps`.
   - Tap to **AUTO**: original detect-and-adapt log line returns.
   - Each tap: `LOADING...` then `Camera ready`.
3. `python diag_camera.py` still reports the negotiated USB speed (unchanged).
