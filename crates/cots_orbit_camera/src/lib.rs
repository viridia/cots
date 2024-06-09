use bevy::{input::mouse::MouseWheel, prelude::*};

/// Tags a camera as capable of orbiting.
#[derive(Component)]
pub struct OrbitCamera {
    pub center: Vec3,
    pub radius: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        OrbitCamera {
            center: Vec3::ZERO,
            radius: 5.0,
            pitch: -0.3,
            yaw: 0.0,
        }
    }
}

pub fn orbit_camera(
    mut ev_wheel: EventReader<MouseWheel>,
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let mut wheel_move = Vec2::ZERO;
    for ev in ev_wheel.read() {
        wheel_move.x += ev.x;
        wheel_move.y += ev.y;
    }

    for (mut pan_orbit, mut transform) in query.iter_mut() {
        pan_orbit.pitch += wheel_move.y * -0.001;
        pan_orbit.yaw += wheel_move.x * -0.001;
        transform.rotation = Quat::from_euler(EulerRot::YXZ, pan_orbit.yaw, pan_orbit.pitch, 0.0);
        let rot_matrix = Mat3::from_quat(transform.rotation);
        transform.translation =
            pan_orbit.center + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, pan_orbit.radius));
    }

    ev_wheel.clear();
}
