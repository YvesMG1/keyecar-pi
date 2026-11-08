"""PlayStation controller input via pygame, turned into one event per control tick."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import ControllerMapping
from .control import STOP_COMMAND, Condition, DriveCommand, apply_deadzone


@dataclass(frozen=True)
class TeleopInputEvent:
    """Controller state for one control tick."""

    command: DriveCommand
    stop_requested: bool = False
    quit_requested: bool = False
    decrease_speed: bool = False
    increase_speed: bool = False
    condition: Condition = Condition.FORWARD
    condition_changed: bool = False

    @property
    def speed_changed(self) -> bool:
        return self.decrease_speed or self.increase_speed


class PlaystationInputAdapter:
    """Reads the left stick and face buttons of a PlayStation controller."""

    def __init__(self, *, mapping: ControllerMapping, deadzone: float):
        self.mapping = mapping
        self.deadzone = deadzone
        import pygame  # only needed on the car, so not a module-level import

        self._pygame = pygame
        self._pygame.init()
        self._pygame.joystick.init()
        self._joystick = None
        # The condition stays latched until another face button is pressed, so
        # straight driving can be recorded without pressing anything.
        self._condition = Condition.FORWARD
        self._was_pressed: dict[int, bool] = {}

    def connect(self):
        """Block until a controller is available and return it."""
        while True:
            self._pygame.joystick.quit()
            self._pygame.joystick.init()
            self._pygame.event.pump()
            if self._pygame.joystick.get_count() > 0:
                self._joystick = self._pygame.joystick.Joystick(0)
                self._joystick.init()
                print(f"Connected: {self._joystick.get_name()}")
                return self._joystick
            time.sleep(1.0)

    def _pressed(self, button: int) -> bool:
        return button < self._joystick.get_numbuttons() and bool(self._joystick.get_button(button))

    def _just_pressed(self, button: int) -> bool:
        """Return True only on the tick a button goes down, not while it is held."""
        pressed = self._pressed(button)
        was_pressed = self._was_pressed.get(button, False)
        self._was_pressed[button] = pressed
        return pressed and not was_pressed

    def read(self) -> TeleopInputEvent:
        """Sample the controller once."""
        try:
            if self._joystick is None or not self._joystick.get_init():
                self.connect()
            self._pygame.event.pump()

            mapping = self.mapping
            command = DriveCommand(
                throttle=apply_deadzone(-self._joystick.get_axis(mapping.left_stick_y_axis), self.deadzone),
                steering=apply_deadzone(self._joystick.get_axis(mapping.left_stick_x_axis), self.deadzone),
            )

            previous_condition = self._condition
            for button, condition in (
                (mapping.forward_button, Condition.FORWARD),
                (mapping.left_button, Condition.LEFT),
                (mapping.right_button, Condition.RIGHT),
            ):
                if self._just_pressed(button):
                    self._condition = condition

            return TeleopInputEvent(
                command=command,
                stop_requested=self._pressed(mapping.stop_button),
                quit_requested=self._pressed(mapping.quit_button),
                decrease_speed=self._just_pressed(mapping.l1_button),
                increase_speed=self._just_pressed(mapping.r1_button),
                condition=self._condition,
                condition_changed=self._condition != previous_condition,
            )
        except self._pygame.error:
            # Fail safe: stop immediately if the controller drops out mid-drive.
            self._joystick = None
            self._was_pressed.clear()
            print("Controller disconnected, stopping until reconnection...")
            return TeleopInputEvent(command=STOP_COMMAND, stop_requested=True)

    def close(self) -> None:
        self._pygame.quit()
