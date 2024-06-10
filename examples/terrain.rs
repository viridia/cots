use cots_orbit_camera::{orbit_camera, OrbitCamera};
use cots_terrain::*;

use bevy::{pbr::ShadowFilteringMethod, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (close_on_esc, create_terrain_meshes, rotate, orbit_camera),
        )
        .run();
}

#[derive(Component)]
struct Shape;

// Setup 3d shapes
fn setup(
    mut commands: Commands,
    // mut meshes: ResMut<Assets<Mesh>>,
    // mut images: ResMut<Assets<Image>>,
    // mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            // intensity: 9000.0,
            intensity: 10000000.0,
            range: 100.,
            shadows_enabled: true,
            shadow_depth_bias: 0.4,
            // shadow_normal_bias: 1.0,
            ..default()
        },
        transform: Transform::from_xyz(8.0, 16.0, 8.0),
        ..default()
    });

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 8., 16.0)
                .looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
            ..default()
        },
        ShadowFilteringMethod::Gaussian,
        OrbitCamera {
            radius: 24.0,
            ..default()
        },
    ));

    commands.spawn(VoxelChunk::new(Vec3::new(0., -8.0, -8.0), 0));
    commands.spawn(VoxelChunk::new(Vec3::new(-32., -8.0, -8.0), 1));
    commands.spawn(VoxelChunk::new(Vec3::new(-32. - 64., -8.0, -8.0), 2));

    // commands.spawn(VoxelChunk {
    //     position: Vec3::new(16.0 - 8.0, -8.0, -8.0),
    //     lod: 0,
    //     status: VoxelChunkStatus::Waiting,
    // });

    // commands.spawn(VoxelChunk {
    //     position: Vec3::new(-16.0 - 8.0, -8.0, -8.0),
    //     lod: 0,
    //     status: VoxelChunkStatus::Waiting,
    // });

    // commands.spawn(VoxelChunk {
    //     position: Vec3::new(-8.0, -8.0, 16.0 - 8.0),
    //     lod: 0,
    //     status: VoxelChunkStatus::Waiting,
    // });

    // commands.spawn(VoxelChunk {
    //     position: Vec3::new(-8.0, -8.0, -16.0 - 8.0),
    //     lod: 0,
    //     status: VoxelChunkStatus::Waiting,
    // });
}

fn rotate(mut query: Query<&mut Transform, With<Shape>>, time: Res<Time>) {
    for mut transform in &mut query {
        transform.rotate_y(time.delta_seconds() / 2.);
    }
}

pub fn close_on_esc(input: Res<ButtonInput<KeyCode>>, mut exit: EventWriter<AppExit>) {
    if input.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}
