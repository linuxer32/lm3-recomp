//! Host controllers and keyboard to the guest's controller
//! (`recomp_services::hid::set_controller`), read on the window thread.
//!
//! Gamepad face buttons map by label by default: Xbox A is Switch A, B is B,
//! X is X, Y is Y. `RECOMP_PAD_LAYOUT=positional` maps by position instead
//! (the bottom button is Switch B, as on a Switch controller).
//!
//! Keyboard: A Enter/Z, B Backspace/X, X C, Y V, L Q, R E, ZL 1, ZR 3,
//! Plus P, Minus M, left stick WASD, right stick IJKL, D-pad arrows.

use recomp_services::hid::{button, ControllerState};
use sdl3::gamepad::{Axis, Button, Gamepad};
use sdl3::keyboard::{KeyboardState, Scancode};

/// Stick full deflection on the guest's scale.
const STICK_MAX: i32 = 0x7FFF;
/// A resting stick that is not centred (worn or drifting) must not keep
/// turning the game's camera: LM3 then runs its view-dependent job work every
/// frame and the frame rate halves. The movement stick keeps a small
/// deadzone; the camera stick uses the standard Xbox right-stick deadzone so
/// a worn stick cannot creep.
const STICK_DEADZONE: i32 = 0x1000;
const CAMERA_DEADZONE: i32 = 0x2000;
/// Trigger travel that counts as ZL/ZR pressed.
const TRIGGER_PRESSED: i16 = 0x4000;

fn positional() -> bool {
    static POSITIONAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *POSITIONAL.get_or_init(|| {
        std::env::var("RECOMP_PAD_LAYOUT").is_ok_and(|layout| layout == "positional")
    })
}

/// A stick's position, with a resting off-centre stick read as centred.
#[cfg(test)]
fn deadzone(x: i32, y: i32) -> (i32, i32) {
    deadzone_with(STICK_DEADZONE, x, y)
}

fn deadzone_with(deadzone: i32, x: i32, y: i32) -> (i32, i32) {
    let length = i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y);
    if length <= i64::from(deadzone) * i64::from(deadzone) {
        (0, 0)
    } else {
        (x, y)
    }
}

/// A gamepad's state.
pub fn gamepad_state(pad: &Gamepad) -> ControllerState {
    let (a, b, x, y) = if positional() {
        (Button::East, Button::South, Button::North, Button::West)
    } else {
        (Button::South, Button::East, Button::West, Button::North)
    };
    let mut buttons = 0;
    for (host, guest) in [
        (a, button::A),
        (b, button::B),
        (x, button::X),
        (y, button::Y),
        (Button::LeftStick, button::STICK_L),
        (Button::RightStick, button::STICK_R),
        (Button::LeftShoulder, button::L),
        (Button::RightShoulder, button::R),
        (Button::Start, button::PLUS),
        (Button::Back, button::MINUS),
        (Button::DPadLeft, button::LEFT),
        (Button::DPadUp, button::UP),
        (Button::DPadRight, button::RIGHT),
        (Button::DPadDown, button::DOWN),
    ] {
        if pad.button(host) {
            buttons |= guest;
        }
    }
    if pad.axis(Axis::TriggerLeft) > TRIGGER_PRESSED {
        buttons |= button::ZL;
    }
    if pad.axis(Axis::TriggerRight) > TRIGGER_PRESSED {
        buttons |= button::ZR;
    }
    // SDL's y axes point down, the guest's up.
    let stick = |x: Axis, y: Axis, deadzone: i32| {
        let scale = |value: i16| (i32::from(value)).clamp(-STICK_MAX, STICK_MAX);
        deadzone_with(deadzone, scale(pad.axis(x)), -scale(pad.axis(y)))
    };
    ControllerState {
        buttons,
        left: stick(Axis::LeftX, Axis::LeftY, STICK_DEADZONE),
        right: stick(Axis::RightX, Axis::RightY, CAMERA_DEADZONE),
    }
}

/// The keyboard's state.
pub fn keyboard_state(keys: &KeyboardState) -> ControllerState {
    let down = |scancode| keys.is_scancode_pressed(scancode);
    let mut buttons = 0;
    for (scancodes, guest) in [
        (&[Scancode::Return, Scancode::Z][..], button::A),
        (&[Scancode::Backspace, Scancode::X], button::B),
        (&[Scancode::C], button::X),
        (&[Scancode::V], button::Y),
        (&[Scancode::Q], button::L),
        (&[Scancode::E], button::R),
        (&[Scancode::_1], button::ZL),
        (&[Scancode::_3], button::ZR),
        (&[Scancode::P], button::PLUS),
        (&[Scancode::M], button::MINUS),
        (&[Scancode::Left], button::LEFT),
        (&[Scancode::Up], button::UP),
        (&[Scancode::Right], button::RIGHT),
        (&[Scancode::Down], button::DOWN),
    ] {
        if scancodes.iter().any(|scancode| down(*scancode)) {
            buttons |= guest;
        }
    }
    let stick = |left: Scancode, up: Scancode, right: Scancode, down_key: Scancode| {
        let axis = |negative: Scancode, positive: Scancode| {
            (i32::from(down(positive)) - i32::from(down(negative))) * STICK_MAX
        };
        (axis(left, right), axis(down_key, up))
    };
    ControllerState {
        buttons,
        left: stick(Scancode::A, Scancode::W, Scancode::D, Scancode::S),
        right: stick(Scancode::J, Scancode::I, Scancode::L, Scancode::K),
    }
}

/// Two sources at once: buttons from either, each stick from whichever is
/// pushed further.
pub fn combine(first: ControllerState, second: ControllerState) -> ControllerState {
    let further = |a: (i32, i32), b: (i32, i32)| {
        let length = |(x, y): (i32, i32)| i64::from(x).pow(2) + i64::from(y).pow(2);
        if length(b) > length(a) {
            b
        } else {
            a
        }
    };
    ControllerState {
        buttons: first.buttons | second.buttons,
        left: further(first.left, second.left),
        right: further(first.right, second.right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_drifting_stick_reads_as_centred() {
        assert_eq!(deadzone(0, 0), (0, 0));
        assert_eq!(deadzone(STICK_DEADZONE, 0), (0, 0));
        assert_eq!(deadzone(-STICK_DEADZONE, 0), (0, 0));
        assert_eq!(deadzone(0, STICK_DEADZONE), (0, 0));
        assert_eq!(deadzone(STICK_DEADZONE + 1, 0), (STICK_DEADZONE + 1, 0));
        assert_eq!(deadzone(-STICK_MAX, 0), (-STICK_MAX, 0));
    }

    #[test]
    fn the_camera_stick_ignores_the_standard_rest_range() {
        assert_eq!(deadzone_with(CAMERA_DEADZONE, 0, 0), (0, 0));
        assert_eq!(deadzone_with(CAMERA_DEADZONE, CAMERA_DEADZONE, 0), (0, 0));
        assert_eq!(
            deadzone_with(CAMERA_DEADZONE, CAMERA_DEADZONE + 1, 0),
            (CAMERA_DEADZONE + 1, 0)
        );
    }

    #[test]
    fn combined_sources_keep_every_button_and_the_further_stick() {
        let pad = ControllerState {
            buttons: button::A,
            left: (100, 0),
            right: (0, -STICK_MAX),
        };
        let keys = ControllerState {
            buttons: button::PLUS,
            left: (0, STICK_MAX),
            right: (0, 0),
        };
        let both = combine(pad, keys);
        assert_eq!(both.buttons, button::A | button::PLUS);
        assert_eq!(both.left, (0, STICK_MAX));
        assert_eq!(both.right, (0, -STICK_MAX));
    }
}
