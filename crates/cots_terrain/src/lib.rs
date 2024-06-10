mod cell_data;
mod chunk;
mod sampler;

use bevy::{color::palettes, prelude::*};

pub use chunk::*;
pub use sampler::*;

pub fn create_terrain_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<&mut VoxelChunk>,
) {
    // TODO: Voxel chunks need a reference to the planetary body they are a part of,
    // and the sampler needs to be assoviated with that.
    let plane = PlaneSampler::new(Vec3::new(0.5, 0.7, 0.3).normalize(), 0.0);
    let sphere = SphereSampler::new(Vec3::new(0., 0., 0.), 4.0);
    let sphere2 = SphereSampler::new(Vec3::new(4., -1., 0.), 2.0);
    let combined = UnionVoxelSampler {
        samplers: &[&plane, &sphere, &sphere2],
    };
    let combined = SubtractiveVoxelSampler {
        pos: &combined,
        neg: &SphereSampler::new(Vec3::new(-2., 3., 0.), 2.5),
    };

    for mut chunk in query.iter_mut() {
        if chunk.status == VoxelChunkStatus::Waiting {
            match combined.solidity(chunk.bounds()) {
                Solidity::Indeterminate => {
                    let mesh = create_voxel_mesh(&combined, chunk.position, chunk.lod);
                    commands.spawn(PbrBundle {
                        mesh: meshes.add(mesh),
                        material: materials.add(Color::from(palettes::css::ROSY_BROWN)),
                        transform: Transform::from_translation(chunk.position),
                        ..default()
                    });
                }
                Solidity::Empty => {}
                Solidity::Solid => {}
            }

            chunk.status = VoxelChunkStatus::Visible;
        }
    }
}
