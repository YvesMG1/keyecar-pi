"""Control loops: PlayStation teleop (data collection) and autonomy with human override.

Both loops use the same command arbitration (see :func:`arbitrate_command`):
stop button > ultrasonic emergency stop > manual stick input > policy output.
That makes every autonomous run a possible DAgger run: whenever the operator
moves the stick, the human command is executed and recorded as an intervention.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from .camera import Camera
from .config import MotorSettings, RuntimeSettings
from .control import ArbitrationResult, DriveCommand, WheelCommand, arbitrate_command
from .inference import PolicyRunner
from .inputs import PlaystationInputAdapter, TeleopInputEvent
from .motor import Motors
from .recording import InferenceDiagnosticsWriter, SessionRecorder
from .ultrasonic import UltrasonicReading, UltrasonicSensor

CONTROLS_HELP = """\
  Face buttons: condition  (Triangle=FORWARD  Square=LEFT  Circle=RIGHT)
  L1 / R1: decrease / increase speed limit
  Cross: stop
  Options: quit"""


@dataclass(frozen=True)
class _Tick:
    """What the operator and the safety sensor say at the start of one control tick."""

    event: TeleopInputEvent
    ultrasonic: UltrasonicReading | None
    speed_limit: int

    @property
    def safety_stop(self) -> bool:
        return bool(self.ultrasonic and self.ultrasonic.emergency_stop)


class _Controls:
    """Controller, ultrasonic sensor, and motors, opened with a safe shutdown order.

    ExitStack callbacks run last-in, first-out: on exit the motors stop first,
    then the sensor releases its pins, then the motor driver cleans up GPIO,
    and the controller is closed last.
    """

    def __init__(self, stack: ExitStack, settings: RuntimeSettings):
        self.settings = settings
        self.speed_limit = settings.motor.default_speed_limit
        self.inputs = PlaystationInputAdapter(mapping=settings.controller, deadzone=settings.teleop.deadzone)
        stack.callback(self.inputs.close)
        self._stack = stack
        self._last_status = 0.0

    def open_hardware(self) -> None:
        """Open the motors and the optional ultrasonic sensor (after the camera and recorder)."""
        self.motors = Motors(settings=self.settings.motor)
        self._stack.callback(self.motors.close)
        self.ultrasonic = None
        if self.settings.ultrasonic.enabled:
            self.ultrasonic = UltrasonicSensor(settings=self.settings.ultrasonic)
            self._stack.callback(self.ultrasonic.close)
        self._stack.callback(self.motors.stop)

    def connect(self, stick_help: str) -> None:
        print("Waiting for PlayStation controller...")
        self.inputs.connect()
        print(f"Controls:\n  Left stick: {stick_help}")
        print(CONTROLS_HELP)

    def read(self) -> _Tick | None:
        """Read the controller and the ultrasonic sensor; None when the operator quits."""
        event = self.inputs.read()
        self.speed_limit = _adjust_speed_limit(self.speed_limit, event, self.settings.motor)
        if event.speed_changed:
            print(f"Speed limit: {self.speed_limit}%")
        if event.condition_changed:
            print(f"Condition: {event.condition.name}")
        if event.quit_requested:
            return None
        reading = self.ultrasonic.read() if self.ultrasonic else None
        return _Tick(event=event, ultrasonic=reading, speed_limit=self.speed_limit)

    def drive(self, decision: ArbitrationResult, tick: _Tick) -> WheelCommand:
        """Send the chosen command to the motors and print a status line now and then."""
        if decision.stop_requested:
            self.motors.stop()
            wheels = WheelCommand(0.0, 0.0)
        else:
            wheels = self.motors.apply_drive(decision.command, tick.speed_limit)
        now = time.monotonic()
        if now - self._last_status >= self.settings.teleop.status_interval_s:
            self._last_status = now
            print(_status_line(tick, decision, wheels))
        return wheels


def _adjust_speed_limit(speed_limit: int, event: TeleopInputEvent, motor_settings: MotorSettings) -> int:
    if event.decrease_speed:
        return max(0, speed_limit - motor_settings.speed_step)
    if event.increase_speed:
        return min(motor_settings.max_duty, speed_limit + motor_settings.speed_step)
    return speed_limit


def _status_line(tick: _Tick, decision: ArbitrationResult, wheels: WheelCommand) -> str:
    command = decision.command
    line = (
        f"speed_limit={tick.speed_limit}% "
        f"throttle={command.throttle:+.2f} steer={command.steering:+.2f} "
        f"left={wheels.left:+.2f} right={wheels.right:+.2f} "
        f"source={decision.source} cond={tick.event.condition.name}"
    )
    reading = tick.ultrasonic
    if reading is not None:
        if reading.filtered_distance_cm is not None:
            line += f" ultra={reading.filtered_distance_cm:5.1f}cm estop={int(reading.emergency_stop)}"
        else:
            line += f" ultra={reading.status.value}"
    return line


def _open_recorder(
    stack: ExitStack,
    settings: RuntimeSettings,
    session_metadata: dict[str, object] | None,
    camera: Camera | None = None,
) -> SessionRecorder:
    recorder = SessionRecorder(
        session_root=settings.teleop.session_root,
        camera_settings=settings.camera,
        metadata=session_metadata,
        camera=camera,
    )
    stack.callback(recorder.close)
    camera_settings = settings.camera
    print(
        f"Recording session: {recorder.session_dir} (camera: {camera_settings.profile} "
        f"{camera_settings.width}x{camera_settings.height}@{camera_settings.fps:g})"
    )
    return recorder


def run_playstation_teleop(
    settings: RuntimeSettings | None = None,
    *,
    record: bool = False,
    session_metadata: dict[str, object] | None = None,
) -> None:
    """Drive with the controller and optionally record a human demonstration."""
    settings = settings or RuntimeSettings()
    with ExitStack() as stack:
        controls = _Controls(stack, settings)
        recorder = _open_recorder(stack, settings, session_metadata) if record else None
        controls.open_hardware()
        controls.connect("drive / steer")

        while (tick := controls.read()) is not None:
            event = tick.event
            decision = arbitrate_command(
                manual=event.command,
                stop_requested=event.stop_requested,
                safety_stop_requested=tick.safety_stop,
            )
            controls.drive(decision, tick)
            if recorder is not None:
                recorder.maybe_capture(
                    human_command=event.command,
                    executed_command=decision.command,
                    speed_limit=tick.speed_limit,
                    stop_requested=event.stop_requested,
                    mode="record",
                    condition=event.condition,
                    control_source=decision.source,
                    ultrasonic_reading=tick.ultrasonic,
                    label_source="human",
                    training_eligible=not event.stop_requested and not tick.safety_stop,
                )
            time.sleep(settings.teleop.loop_delay_s)


def run_autonomy_loop(
    settings: RuntimeSettings | None = None,
    *,
    model_path: str | Path,
    metadata_path: str | Path,
    record: bool = False,
    session_metadata: dict[str, object] | None = None,
    ema_alpha: float = 0.0,
    diagnostics: bool = False,
) -> None:
    """Drive with a trained policy while the operator keeps the controller.

    Any stick movement takes over until the stick is released. Each takeover
    gets an ``intervention_id`` in the recording, so it can later be turned into
    DAgger training targets. During a takeover the policy keeps observing frames
    so its frame history stays continuous.
    """
    settings = settings or RuntimeSettings()
    with ExitStack() as stack:
        controls = _Controls(stack, settings)
        runner = PolicyRunner(
            model_path=model_path, metadata_path=metadata_path, ema_alpha=ema_alpha, diagnostics=diagnostics
        )
        print(f"Policy: {runner.metadata.variant_name} ({model_path})")
        camera = Camera(settings=settings.camera)
        stack.callback(camera.close)
        # The recorder shares the inference camera: Picamera2 does not reliably
        # support a second client on the same device.
        recorder = _open_recorder(stack, settings, session_metadata, camera=camera) if record else None

        diagnostics_writer = None
        if diagnostics:
            if recorder is not None:
                diagnostics_writer = InferenceDiagnosticsWriter(recorder.session_dir / "diagnostics")
            else:
                diagnostics_writer = InferenceDiagnosticsWriter.in_new_folder(settings.teleop.session_root)
            stack.callback(diagnostics_writer.close)
            print(f"Inference diagnostics: {diagnostics_writer.output_dir}")

        controls.open_hardware()
        controls.connect("manual override (any deflection takes over)")

        previous_command = (0.0, 0.0)
        was_manual = False
        intervention_count = 0
        intervention_id = ""

        while (tick := controls.read()) is not None:
            event = tick.event
            manual = not event.command.is_zero()
            if manual and not was_manual:
                intervention_count += 1
                intervention_id = f"intv_{intervention_count:06d}"
            elif not manual:
                intervention_id = ""

            policy_command: DriveCommand | None = None
            image = None
            camera_metadata = None
            if event.stop_requested or tick.safety_stop:
                # A stop breaks the frame history. Takeovers keep it via observe().
                runner.reset()
            else:
                if recorder is not None:
                    image, camera_metadata = camera.capture_with_metadata()
                else:
                    image = camera.capture()
                if manual:
                    if not was_manual:
                        runner.reset_output_smoothing()
                    runner.observe(image=image, condition=event.condition)
                else:
                    policy_command = runner.predict(
                        image=image, motor_state=previous_command, condition=event.condition
                    )

            decision = arbitrate_command(
                manual=event.command if manual else None,
                autonomy=policy_command,
                stop_requested=event.stop_requested,
                safety_stop_requested=tick.safety_stop,
            )
            controls.drive(decision, tick)
            # The policy's previous-command input is what was actually executed.
            previous_command = (decision.command.throttle, decision.command.steering)

            recorded_frame = None
            if recorder is not None:
                if manual:
                    label_source = "human"
                elif policy_command is not None:
                    label_source = "model"
                else:
                    label_source = "none"
                recorded_frame = recorder.maybe_capture(
                    human_command=event.command if manual else None,
                    model_command=policy_command,
                    executed_command=decision.command,
                    speed_limit=tick.speed_limit,
                    stop_requested=event.stop_requested,
                    mode="autonomy_active",
                    condition=event.condition,
                    control_source=decision.source,
                    ultrasonic_reading=tick.ultrasonic,
                    intervention_state="manual_override" if manual else "none",
                    intervention_id=intervention_id,
                    label_source=label_source,
                    training_eligible=manual and not event.stop_requested and not tick.safety_stop,
                    image=image,
                    camera_metadata=camera_metadata,
                    force=diagnostics and policy_command is not None,
                )

            if diagnostics_writer is not None and policy_command is not None:
                timing = runner.last_diagnostics
                diagnostics_writer.write(timing, frame_id=recorded_frame.frame_id if recorded_frame else None)
                print(f"latency total={timing.total_ms:.2f}ms policy={timing.policy_onnx_ms:.2f}ms")

            was_manual = manual
            # No sleep: camera capture and inference already pace the loop, and
            # an extra delay would only add latency.
