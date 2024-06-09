use cots_orbit_camera::{orbit_camera, OrbitCamera};

use bevy::{
    color::palettes,
    math::bounding::{Aabb3d, BoundingSphere, IntersectsVolume},
    pbr::ShadowFilteringMethod,
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
    utils::HashMap,
};

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

    commands.spawn(VoxelChunk {
        position: Vec3::new(0., -8.0, -8.0),
        lod: 0,
        status: VoxelChunkStatus::Waiting,
    });

    commands.spawn(VoxelChunk {
        position: Vec3::new(-32., -8.0, -8.0),
        lod: 1,
        status: VoxelChunkStatus::Waiting,
    });

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

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum VoxelChunkStatus {
    /// Chunk is currently visible
    Visible = 0,

    /// Chunk is visible, but a pending replacement is being generated.
    Stale,

    /// Chunk is not visible, waiting for async generation.
    #[default]
    Waiting,

    /// Chunk is not visible, still being generated.
    Generating,

    /// Chunk is generated, waiting to replace the stale chunk(s).
    Pending,
}

#[derive(Default, Debug, Clone, Component)]
pub struct VoxelChunk {
    /// Position of the chunk in object space (planetary coordinates).
    position: Vec3,
    /// Level of details, represented a power of 2 scale. Level 0 is 1 meter units, Level 1
    /// is 2 meter units, etc. Can be negative for sub-meter precision.
    lod: i32,

    /// Current status of the chunk.
    status: VoxelChunkStatus,
}

impl VoxelChunk {
    pub fn new(position: Vec3, lod: i32) -> Self {
        Self {
            position,
            lod,
            status: VoxelChunkStatus::Waiting,
        }
    }

    /// Return the bounding box of the chunk.
    pub fn bounds(&self) -> Aabb3d {
        let size = Vec3::splat(8.0 * 2.0f32.powi(self.lod));
        Aabb3d::new(
            Vec3::new(self.position.x, self.position.y, self.position.z) + size,
            size,
        )
    }
}

// A 19 x 19 x 19 grid of voxels
// Used to constuct a 16x16x16 terrain mesh, with skirt needed to compute derivatives.
struct VoxelSampleGrid {}

impl VoxelSampleGrid {
    /// TODO: Move this constant to CHUNK_SIZE.
    /// Number of samples in each dimension.
    const SAMPLE_COUNT: usize = 19;
}

/// A value sampled from a location in 3d space which indicates whether the location is solid
/// or empty, and what kind of material is present.
#[derive(Debug, Clone, Copy)]
pub struct VoxelSampleValue {
    /// Distance to the surface (in SDF fashion), negative values are inside the surface.
    pub distance: f32,
    // TODO: Material, including "holes".
    // primary material.
    // secondary material and weight.
    // tertiary material and weight.
}

/// Result code that specifies whether a 2d region is empty, solid, or indeterminate.
pub enum Solidity {
    Indeterminate = 0,
    Empty,
    Solid,
}

trait VoxelSampler {
    /// Returns the distance to the surface at the given position. Negative values mean that
    /// we are inside the solid region.
    fn distance(&self, pos: Vec3) -> f32 {
        self.sample(pos).distance
    }

    /// Return information about the distance and material at the given position.
    fn sample(&self, pos: Vec3) -> VoxelSampleValue;

    /// Returns whether the given region is solid, empty, or indeterminate.
    fn solidity(&self, bounds: Aabb3d) -> Solidity;
}

/// A sampler that additively combines the outputs of multiple samplers.
struct UnionVoxelSampler<'a> {
    samplers: &'a [&'a dyn VoxelSampler],
}

impl<'a> VoxelSampler for UnionVoxelSampler<'a> {
    fn sample(&self, pos: Vec3) -> VoxelSampleValue {
        let mut min_dist = f32::INFINITY;
        for sampler in self.samplers {
            let dist = sampler.distance(pos);
            min_dist = min_dist.min(dist);
        }
        VoxelSampleValue { distance: min_dist }
    }

    fn solidity(&self, bounds: Aabb3d) -> Solidity {
        let mut empty = true;
        for sampler in self.samplers {
            match sampler.solidity(bounds) {
                Solidity::Indeterminate => {
                    // If any sampler is indeterminate, the region is not empty, but may be
                    // either solid or indeterminate.
                    empty = false;
                }
                Solidity::Empty => {}
                Solidity::Solid => {
                    // If any sampler is solid, the region is solid.
                    return Solidity::Solid;
                }
            }
        }
        if empty {
            Solidity::Empty
        } else {
            Solidity::Indeterminate
        }
    }
}

struct SubtractiveVoxelSampler<'a> {
    pos: &'a dyn VoxelSampler,
    neg: &'a dyn VoxelSampler,
}

impl VoxelSampler for SubtractiveVoxelSampler<'_> {
    fn sample(&self, pos: Vec3) -> VoxelSampleValue {
        let pos_dist = self.pos.sample(pos).distance;
        let neg_dist = -self.neg.sample(pos).distance;
        VoxelSampleValue {
            distance: pos_dist.max(neg_dist),
        }
    }

    fn solidity(&self, bounds: Aabb3d) -> Solidity {
        let pos_solid = self.pos.solidity(bounds);
        let neg_solid = self.neg.solidity(bounds);
        match (pos_solid, neg_solid) {
            // If the negative region is empty, then it's whatever the positive region is.
            (pos, Solidity::Empty) => pos,
            // If the positive region is emoty, then it's empty.
            (Solidity::Empty, _) => Solidity::Empty,
            // If the negative region is solid, then it's empty.
            (_, Solidity::Solid) => Solidity::Empty,
            // Otherwise, we don't know.
            _ => Solidity::Indeterminate,
        }
    }
}

struct PlaneSampler {
    normal: Vec3,
    offset: f32,
}

impl PlaneSampler {
    fn new(normal: Vec3, offset: f32) -> Self {
        Self { normal, offset }
    }
}

impl VoxelSampler for PlaneSampler {
    fn sample(&self, pos: Vec3) -> VoxelSampleValue {
        let dist = pos.dot(self.normal) + self.offset;
        VoxelSampleValue { distance: dist }
    }

    fn solidity(&self, bounds: Aabb3d) -> Solidity {
        let mut empty = true;
        let mut solid = true;
        for x in [bounds.min.x, bounds.max.x] {
            for y in [bounds.min.y, bounds.max.y] {
                for z in [bounds.min.z, bounds.max.z] {
                    let pos = Vec3::new(x, y, z);
                    let dist = pos.dot(self.normal) + self.offset;
                    if dist < 0.0 {
                        solid = false;
                    } else if dist > 0.0 {
                        empty = false;
                    }
                    if !solid && !empty {
                        return Solidity::Indeterminate;
                    }
                }
            }
        }
        if empty {
            Solidity::Empty
        } else if solid {
            Solidity::Solid
        } else {
            Solidity::Indeterminate
        }
    }
}

struct SphereSampler {
    center: Vec3,
    radius: f32,
}

impl SphereSampler {
    fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl VoxelSampler for SphereSampler {
    fn sample(&self, pos: Vec3) -> VoxelSampleValue {
        let dist = (pos - self.center).length() - self.radius;
        VoxelSampleValue { distance: dist }
    }

    fn solidity(&self, bounds: Aabb3d) -> Solidity {
        // If sphere is outside of any plane, it's empty.
        if self.center.x + self.radius <= bounds.min.x
            || self.center.x - self.radius >= bounds.max.x
            || self.center.y + self.radius <= bounds.min.y
            || self.center.y - self.radius >= bounds.max.y
            || self.center.z + self.radius <= bounds.min.z
            || self.center.z - self.radius >= bounds.max.z
        {
            return Solidity::Empty;
        }

        // If box corners are entirely within the sphere, it's solid.
        let mut solid = true;
        let radius_sq = self.radius * self.radius;
        for x in [bounds.min.x, bounds.max.x] {
            for y in [bounds.min.y, bounds.max.y] {
                for z in [bounds.min.z, bounds.max.z] {
                    let pos = Vec3::new(x, y, z);
                    let dist_sq = (pos - self.center).length_squared();
                    if dist_sq > radius_sq {
                        solid = false;
                    }
                }
            }
        }
        if solid {
            Solidity::Solid
        } else if bounds.intersects(&BoundingSphere::new(self.center, self.radius)) {
            Solidity::Indeterminate
        } else {
            Solidity::Empty
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CellVertex {
    position: Vec3,
    normal: Vec3,
    triangle_count: usize,
    mapped_index: u32,
}

impl CellVertex {
    fn new(position: Vec3) -> Self {
        Self {
            position,
            normal: Vec3::ZERO,
            triangle_count: 0,
            mapped_index: 0,
        }
    }
}

fn create_terrain_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<&mut VoxelChunk>,
) {
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
        // How do we get the sampler?
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

#[allow(clippy::needless_range_loop)]
fn create_voxel_mesh(sampler: &dyn VoxelSampler, origin: Vec3, lod: i32) -> Mesh {
    let scale = 2.0f32.powi(lod);
    let mut vertices: Vec<CellVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // let mut position_offsets: [[[[usize; 4]; 18]; 18]; 18] = [[[[0; 4]; 18]; 18]; 18];
    let mut reused_vertices: HashMap<UVec3, usize> =
        HashMap::with_capacity(VoxelSampleGrid::SAMPLE_COUNT * VoxelSampleGrid::SAMPLE_COUNT * 4);

    // let center = Vec3::new(9.0, 9.0, 9.0);
    for x in 0..VoxelSampleGrid::SAMPLE_COUNT - 1 {
        for y in 0..VoxelSampleGrid::SAMPLE_COUNT - 1 {
            for z in 0..VoxelSampleGrid::SAMPLE_COUNT - 1 {
                let mut cube_case: usize = 0;
                let mut corner_samples: [f32; 8] = [0.0; 8];
                for i in 0..8 {
                    let dx = i & 1;
                    let dy = (i >> 1) & 1;
                    let dz = (i >> 2) & 1;
                    corner_samples[i] = sampler
                        .sample(
                            Vec3::new(
                                (x + dx) as f32 - 1.0,
                                (y + dy) as f32 - 1.0,
                                (z + dz) as f32 - 1.0,
                            ) * scale
                                + origin,
                        )
                        .distance;
                    // corner_samples[i] = grid.samples[x + dx][y + dy][z + dz];
                    if corner_samples[i] < 0.0 {
                        cube_case |= 1 << i;
                    }
                }

                let rclass = &REGULAR_CELL_DATA[REGULAR_CELL_CLASS[cube_case] as usize];
                if rclass.vertex_count == 0 {
                    continue;
                }

                // let reuse_mask =
                //     (x > 0) as usize | ((y > 0) as usize) << 1 | ((z > 0) as usize) << 2;

                let cell_origin = UVec3::new((x << 9) as u32, (y << 9) as u32, (z << 9) as u32);

                let rvtx = REGULAR_VERTEX_DATA[cube_case];
                let mut cell_indices: [usize; 16] = [0; 16];
                let mut cell_vertices: [Vec3; 16] = [Vec3::ZERO; 16];
                for (i, vdata) in rvtx.iter().enumerate() {
                    let n0 = ((vdata >> 4) & 0x0F) as usize;
                    let n1 = (vdata & 0x0F) as usize;
                    let u0 = cell_origin + corner_offset(n0 as u32) * 512;
                    let u1 = cell_origin + corner_offset(n1 as u32) * 512;
                    let s0 = corner_samples[n0];
                    let s1 = corner_samples[n1];
                    let t = (512.0 * s1 / (s1 - s0)) as u32;
                    // let v_index = ((vdata >> 8) & 0x0f) as usize;
                    // let v_reuse = (vdata >> 12) as usize;
                    // let mut try_reuse = false;
                    let pos = (t * u0 + (512 - t) * u1) / 512;
                    // if t & 0xFF != 0 {
                    //     pos = (t * u0 + (0x100 - t) * u1) / 256;
                    //     // try_reuse = true;
                    // } else if t == 0 {
                    //     pos = u1;
                    //     if n0 != 7 {
                    //         // owned
                    //     } else {
                    //         // v_reuse = 0;
                    //         // try_reuse = true;
                    //     }
                    // } else {
                    //     pos = u0;
                    //     // try_reuse = true;
                    // }

                    match reused_vertices.get(&pos) {
                        Some(&offs) => {
                            cell_indices[i] = offs;
                            cell_vertices[i] = vertices[offs].position;
                        }
                        None => {
                            let fpos = Vec3::new(
                                pos.x as f32 / 512.0 - 1.0,
                                pos.y as f32 / 512.0 - 1.0,
                                pos.z as f32 / 512.0 - 1.0,
                            ) * scale;
                            let ioffset = vertices.len();
                            cell_indices[i] = ioffset;
                            cell_vertices[i] = fpos;
                            reused_vertices.insert(pos, ioffset);
                            vertices.push(CellVertex::new(fpos));
                        }
                    }

                    // More efficient method of reusing vertices, but which has a subtle
                    // bug that I have not been able to track down.

                    // Flag indicates we want to re-used a vertex.
                    // if try_reuse && (v_reuse & reuse_mask) == v_reuse {
                    //     let px = x - (v_reuse & 0x01);
                    //     let py = y - ((v_reuse >> 1) & 0x01);
                    //     let pz = z - ((v_reuse >> 2) & 0x01);
                    //     let offs = position_offsets[px][py][pz][v_index];
                    //     cell_indices[i] = offs;
                    //     cell_vertices[i] = vertices[offs].position;
                    // } else {
                    //     // let t = t as f32 / 256.0;
                    //     // let pos = t * p0 + (1.0 - t) * p1 - center;
                    //     let fpos =
                    //         Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32) / 256.0 - center;
                    //     let ioffset = vertices.len();
                    //     cell_indices[i] = ioffset;
                    //     cell_vertices[i] = fpos;
                    //     position_offsets[x][y][z][v_index] = ioffset;
                    //     vertices.push(CellVertex::new(fpos));
                    // }
                }

                let num_indices = rclass.vertex_indices.len();
                let num_triangles = num_indices / 3;
                for i in 0..num_triangles {
                    let i0 = rclass.vertex_indices[i * 3] as u32;
                    let i1 = rclass.vertex_indices[i * 3 + 1] as u32;
                    let i2 = rclass.vertex_indices[i * 3 + 2] as u32;
                    let c0 = cell_indices[i0 as usize];
                    let c1 = cell_indices[i1 as usize];
                    let c2 = cell_indices[i2 as usize];
                    let v0 = cell_vertices[i0 as usize];
                    let v1 = cell_vertices[i1 as usize];
                    let v2 = cell_vertices[i2 as usize];
                    let tri = Triangle3d::new(v0, v1, v2);
                    if tri.is_degenerate() {
                        continue;
                    }
                    // Triangles contribute to the normal calculation even if they are
                    // in the margin.
                    let n = tri.normal().unwrap().as_vec3();
                    let v0 = &mut vertices[c0];
                    v0.normal += n;
                    v0.triangle_count += 1;
                    let v1 = &mut vertices[c1];
                    v1.normal += n;
                    v1.triangle_count += 1;
                    let v2 = &mut vertices[c2];
                    v2.normal += n;
                    v2.triangle_count += 1;
                    // Only push indices if the cell is not in the margins.
                    if x > 0 && y > 0 && z > 0 && x < 17 && y < 17 && z < 17 {
                        indices.push(c0 as u32);
                        indices.push(c1 as u32);
                        indices.push(c2 as u32);
                    }
                }
            }
        }
    }

    let num_vertices_used = vertices.iter().filter(|v| v.triangle_count > 0).count();
    let mut position = vec![Vec3::ZERO; num_vertices_used];
    let mut normal = vec![Vec3::ZERO; num_vertices_used];
    let mut mapped_index: u32 = 0;
    for v in vertices.iter_mut() {
        if v.triangle_count == 0 {
            continue;
        }
        v.mapped_index = mapped_index;
        normal[mapped_index as usize] = v.normal / v.triangle_count as f32;
        position[mapped_index as usize] = v.position;
        mapped_index += 1;
    }

    for ind in indices.iter_mut() {
        *ind = vertices[*ind as usize].mapped_index;
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, position);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normal);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_aabb();
    mesh
}

/// Given a corner index (0..7), return the offset of the corner in the cell.
fn corner_offset(n: u32) -> UVec3 {
    UVec3::new(n & 1, (n >> 1) & 1, (n >> 2) & 1)
}

// Tables taken from https://github.com/EricLengyel/Transvoxel/blob/main/Transvoxel.cpp

struct RegularCellData {
    vertex_count: u8,
    vertex_indices: &'static [u8],
}

impl RegularCellData {
    const fn new(vertex_count: u8, vertex_indices: &'static [u8]) -> Self {
        Self {
            vertex_count,
            vertex_indices,
        }
    }
}

// The `REGULAR_CELL_CLASS` table maps an 8-bit regular Marching Cubes case index to
// an equivalence class index. Even though there are 18 equivalence classes in our
// modified Marching Cubes algorithm, a couple of them use the same exact triangulations,
// just with different vertex locations. We combined those classes for this table so
// that the class index ranges from 0 to 15.

const REGULAR_CELL_CLASS: [u8; 256] = [
    0x00, 0x01, 0x01, 0x03, 0x01, 0x03, 0x02, 0x04, 0x01, 0x02, 0x03, 0x04, 0x03, 0x04, 0x04, 0x03,
    0x01, 0x03, 0x02, 0x04, 0x02, 0x04, 0x06, 0x0C, 0x02, 0x05, 0x05, 0x0B, 0x05, 0x0A, 0x07, 0x04,
    0x01, 0x02, 0x03, 0x04, 0x02, 0x05, 0x05, 0x0A, 0x02, 0x06, 0x04, 0x0C, 0x05, 0x07, 0x0B, 0x04,
    0x03, 0x04, 0x04, 0x03, 0x05, 0x0B, 0x07, 0x04, 0x05, 0x07, 0x0A, 0x04, 0x08, 0x0E, 0x0E, 0x03,
    0x01, 0x02, 0x02, 0x05, 0x03, 0x04, 0x05, 0x0B, 0x02, 0x06, 0x05, 0x07, 0x04, 0x0C, 0x0A, 0x04,
    0x03, 0x04, 0x05, 0x0A, 0x04, 0x03, 0x07, 0x04, 0x05, 0x07, 0x08, 0x0E, 0x0B, 0x04, 0x0E, 0x03,
    0x02, 0x06, 0x05, 0x07, 0x05, 0x07, 0x08, 0x0E, 0x06, 0x09, 0x07, 0x0F, 0x07, 0x0F, 0x0E, 0x0D,
    0x04, 0x0C, 0x0B, 0x04, 0x0A, 0x04, 0x0E, 0x03, 0x07, 0x0F, 0x0E, 0x0D, 0x0E, 0x0D, 0x02, 0x01,
    0x01, 0x02, 0x02, 0x05, 0x02, 0x05, 0x06, 0x07, 0x03, 0x05, 0x04, 0x0A, 0x04, 0x0B, 0x0C, 0x04,
    0x02, 0x05, 0x06, 0x07, 0x06, 0x07, 0x09, 0x0F, 0x05, 0x08, 0x07, 0x0E, 0x07, 0x0E, 0x0F, 0x0D,
    0x03, 0x05, 0x04, 0x0B, 0x05, 0x08, 0x07, 0x0E, 0x04, 0x07, 0x03, 0x04, 0x0A, 0x0E, 0x04, 0x03,
    0x04, 0x0A, 0x0C, 0x04, 0x07, 0x0E, 0x0F, 0x0D, 0x0B, 0x0E, 0x04, 0x03, 0x0E, 0x02, 0x0D, 0x01,
    0x03, 0x05, 0x05, 0x08, 0x04, 0x0A, 0x07, 0x0E, 0x04, 0x07, 0x0B, 0x0E, 0x03, 0x04, 0x04, 0x03,
    0x04, 0x0B, 0x07, 0x0E, 0x0C, 0x04, 0x0F, 0x0D, 0x0A, 0x0E, 0x0E, 0x02, 0x04, 0x03, 0x0D, 0x01,
    0x04, 0x07, 0x0A, 0x0E, 0x0B, 0x0E, 0x0E, 0x02, 0x0C, 0x0F, 0x04, 0x0D, 0x04, 0x0D, 0x03, 0x01,
    0x03, 0x04, 0x04, 0x03, 0x04, 0x03, 0x0D, 0x01, 0x04, 0x0D, 0x03, 0x01, 0x03, 0x01, 0x01, 0x00,
];

// The `REGULAR_CELL_DATA` table holds the triangulation data for all 16 distinct classes to
// which a case can be mapped by the `REGULAR_CELL_CLASS` table.

const REGULAR_CELL_DATA: [RegularCellData; 16] = [
    RegularCellData::new(0x0, &[]),
    RegularCellData::new(0x3, &[0, 1, 2]),
    RegularCellData::new(0x6, &[0, 1, 2, 3, 4, 5]),
    RegularCellData::new(0x4, &[0, 1, 2, 0, 2, 3]),
    RegularCellData::new(0x5, &[0, 1, 4, 1, 3, 4, 1, 2, 3]),
    RegularCellData::new(0x7, &[0, 1, 2, 0, 2, 3, 4, 5, 6]),
    RegularCellData::new(0x9, &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
    RegularCellData::new(0x8, &[0, 1, 4, 1, 3, 4, 1, 2, 3, 5, 6, 7]),
    RegularCellData::new(0x8, &[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]),
    RegularCellData::new(0xC, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    RegularCellData::new(0x6, &[0, 4, 5, 0, 1, 4, 1, 3, 4, 1, 2, 3]),
    RegularCellData::new(0x6, &[0, 5, 4, 0, 4, 1, 1, 4, 3, 1, 3, 2]),
    RegularCellData::new(0x6, &[0, 4, 5, 0, 3, 4, 0, 1, 3, 1, 2, 3]),
    RegularCellData::new(0x6, &[0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5]),
    RegularCellData::new(0x7, &[0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6]),
    RegularCellData::new(0x9, &[0, 4, 5, 0, 3, 4, 0, 1, 3, 1, 2, 3, 6, 7, 8]),
];

// The `regular_vertex_data` table gives the vertex locations for every one of the 256 possible
// cases in the modified Marching Cubes algorithm. Each 16-bit value also provides information
// about whether a vertex can be reused from a neighboring cell. See Section 3.3 for details.
// The low byte contains the indexes for the two endpoints of the edge on which the vertex lies,
// as numbered in Figure 3.7. The high byte contains the vertex reuse data shown in Figure 3.8.

const REGULAR_VERTEX_DATA: [&[u16]; 256] = [
    &[],
    &[0x6201, 0x5102, 0x3304],
    &[0x6201, 0x2315, 0x4113],
    &[0x5102, 0x3304, 0x2315, 0x4113],
    &[0x5102, 0x4223, 0x1326],
    &[0x3304, 0x6201, 0x4223, 0x1326],
    &[0x6201, 0x2315, 0x4113, 0x5102, 0x4223, 0x1326],
    &[0x4223, 0x1326, 0x3304, 0x2315, 0x4113],
    &[0x4113, 0x8337, 0x4223],
    &[0x6201, 0x5102, 0x3304, 0x4223, 0x4113, 0x8337],
    &[0x6201, 0x2315, 0x8337, 0x4223],
    &[0x5102, 0x3304, 0x2315, 0x8337, 0x4223],
    &[0x5102, 0x4113, 0x8337, 0x1326],
    &[0x4113, 0x8337, 0x1326, 0x3304, 0x6201],
    &[0x6201, 0x2315, 0x8337, 0x1326, 0x5102],
    &[0x3304, 0x2315, 0x8337, 0x1326],
    &[0x3304, 0x1146, 0x2245],
    &[0x6201, 0x5102, 0x1146, 0x2245],
    &[0x6201, 0x2315, 0x4113, 0x3304, 0x1146, 0x2245],
    &[0x2315, 0x4113, 0x5102, 0x1146, 0x2245],
    &[0x5102, 0x4223, 0x1326, 0x3304, 0x1146, 0x2245],
    &[0x1146, 0x2245, 0x6201, 0x4223, 0x1326],
    &[
        0x3304, 0x1146, 0x2245, 0x6201, 0x2315, 0x4113, 0x5102, 0x4223, 0x1326,
    ],
    &[0x4223, 0x1326, 0x1146, 0x2245, 0x2315, 0x4113],
    &[0x4223, 0x4113, 0x8337, 0x3304, 0x1146, 0x2245],
    &[0x6201, 0x5102, 0x1146, 0x2245, 0x4223, 0x4113, 0x8337],
    &[0x4223, 0x6201, 0x2315, 0x8337, 0x3304, 0x1146, 0x2245],
    &[0x4223, 0x8337, 0x2315, 0x2245, 0x1146, 0x5102],
    &[0x5102, 0x4113, 0x8337, 0x1326, 0x3304, 0x1146, 0x2245],
    &[0x4113, 0x8337, 0x1326, 0x1146, 0x2245, 0x6201],
    &[
        0x6201, 0x2315, 0x8337, 0x1326, 0x5102, 0x3304, 0x1146, 0x2245,
    ],
    &[0x2245, 0x2315, 0x8337, 0x1326, 0x1146],
    &[0x2315, 0x2245, 0x8157],
    &[0x6201, 0x5102, 0x3304, 0x2315, 0x2245, 0x8157],
    &[0x4113, 0x6201, 0x2245, 0x8157],
    &[0x2245, 0x8157, 0x4113, 0x5102, 0x3304],
    &[0x5102, 0x4223, 0x1326, 0x2315, 0x2245, 0x8157],
    &[0x6201, 0x4223, 0x1326, 0x3304, 0x2315, 0x2245, 0x8157],
    &[0x6201, 0x2245, 0x8157, 0x4113, 0x5102, 0x4223, 0x1326],
    &[0x4223, 0x1326, 0x3304, 0x2245, 0x8157, 0x4113],
    &[0x4223, 0x4113, 0x8337, 0x2315, 0x2245, 0x8157],
    &[
        0x6201, 0x5102, 0x3304, 0x4223, 0x4113, 0x8337, 0x2315, 0x2245, 0x8157,
    ],
    &[0x8337, 0x4223, 0x6201, 0x2245, 0x8157],
    &[0x5102, 0x3304, 0x2245, 0x8157, 0x8337, 0x4223],
    &[0x5102, 0x4113, 0x8337, 0x1326, 0x2315, 0x2245, 0x8157],
    &[
        0x4113, 0x8337, 0x1326, 0x3304, 0x6201, 0x2315, 0x2245, 0x8157,
    ],
    &[0x5102, 0x1326, 0x8337, 0x8157, 0x2245, 0x6201],
    &[0x8157, 0x8337, 0x1326, 0x3304, 0x2245],
    &[0x2315, 0x3304, 0x1146, 0x8157],
    &[0x6201, 0x5102, 0x1146, 0x8157, 0x2315],
    &[0x3304, 0x1146, 0x8157, 0x4113, 0x6201],
    &[0x4113, 0x5102, 0x1146, 0x8157],
    &[0x2315, 0x3304, 0x1146, 0x8157, 0x5102, 0x4223, 0x1326],
    &[0x1326, 0x4223, 0x6201, 0x2315, 0x8157, 0x1146],
    &[
        0x3304, 0x1146, 0x8157, 0x4113, 0x6201, 0x5102, 0x4223, 0x1326,
    ],
    &[0x1326, 0x1146, 0x8157, 0x4113, 0x4223],
    &[0x2315, 0x3304, 0x1146, 0x8157, 0x4223, 0x4113, 0x8337],
    &[
        0x6201, 0x5102, 0x1146, 0x8157, 0x2315, 0x4223, 0x4113, 0x8337,
    ],
    &[0x3304, 0x1146, 0x8157, 0x8337, 0x4223, 0x6201],
    &[0x4223, 0x5102, 0x1146, 0x8157, 0x8337],
    &[
        0x2315, 0x3304, 0x1146, 0x8157, 0x5102, 0x4113, 0x8337, 0x1326,
    ],
    &[0x6201, 0x4113, 0x8337, 0x1326, 0x1146, 0x8157, 0x2315],
    &[0x6201, 0x3304, 0x1146, 0x8157, 0x8337, 0x1326, 0x5102],
    &[0x1326, 0x1146, 0x8157, 0x8337],
    &[0x1326, 0x8267, 0x1146],
    &[0x6201, 0x5102, 0x3304, 0x1326, 0x8267, 0x1146],
    &[0x6201, 0x2315, 0x4113, 0x1326, 0x8267, 0x1146],
    &[0x5102, 0x3304, 0x2315, 0x4113, 0x1326, 0x8267, 0x1146],
    &[0x5102, 0x4223, 0x8267, 0x1146],
    &[0x3304, 0x6201, 0x4223, 0x8267, 0x1146],
    &[0x5102, 0x4223, 0x8267, 0x1146, 0x6201, 0x2315, 0x4113],
    &[0x1146, 0x8267, 0x4223, 0x4113, 0x2315, 0x3304],
    &[0x4113, 0x8337, 0x4223, 0x1326, 0x8267, 0x1146],
    &[
        0x6201, 0x5102, 0x3304, 0x4223, 0x4113, 0x8337, 0x1326, 0x8267, 0x1146,
    ],
    &[0x6201, 0x2315, 0x8337, 0x4223, 0x1326, 0x8267, 0x1146],
    &[
        0x5102, 0x3304, 0x2315, 0x8337, 0x4223, 0x1326, 0x8267, 0x1146,
    ],
    &[0x8267, 0x1146, 0x5102, 0x4113, 0x8337],
    &[0x6201, 0x4113, 0x8337, 0x8267, 0x1146, 0x3304],
    &[0x6201, 0x2315, 0x8337, 0x8267, 0x1146, 0x5102],
    &[0x1146, 0x3304, 0x2315, 0x8337, 0x8267],
    &[0x3304, 0x1326, 0x8267, 0x2245],
    &[0x1326, 0x8267, 0x2245, 0x6201, 0x5102],
    &[0x3304, 0x1326, 0x8267, 0x2245, 0x6201, 0x2315, 0x4113],
    &[0x1326, 0x8267, 0x2245, 0x2315, 0x4113, 0x5102],
    &[0x5102, 0x4223, 0x8267, 0x2245, 0x3304],
    &[0x6201, 0x4223, 0x8267, 0x2245],
    &[
        0x5102, 0x4223, 0x8267, 0x2245, 0x3304, 0x6201, 0x2315, 0x4113,
    ],
    &[0x4113, 0x4223, 0x8267, 0x2245, 0x2315],
    &[0x3304, 0x1326, 0x8267, 0x2245, 0x4223, 0x4113, 0x8337],
    &[
        0x1326, 0x8267, 0x2245, 0x6201, 0x5102, 0x4223, 0x4113, 0x8337,
    ],
    &[
        0x3304, 0x1326, 0x8267, 0x2245, 0x4223, 0x6201, 0x2315, 0x8337,
    ],
    &[0x5102, 0x1326, 0x8267, 0x2245, 0x2315, 0x8337, 0x4223],
    &[0x3304, 0x2245, 0x8267, 0x8337, 0x4113, 0x5102],
    &[0x8337, 0x8267, 0x2245, 0x6201, 0x4113],
    &[0x5102, 0x6201, 0x2315, 0x8337, 0x8267, 0x2245, 0x3304],
    &[0x2315, 0x8337, 0x8267, 0x2245],
    &[0x2315, 0x2245, 0x8157, 0x1326, 0x8267, 0x1146],
    &[
        0x6201, 0x5102, 0x3304, 0x2315, 0x2245, 0x8157, 0x1326, 0x8267, 0x1146,
    ],
    &[0x6201, 0x2245, 0x8157, 0x4113, 0x1326, 0x8267, 0x1146],
    &[
        0x2245, 0x8157, 0x4113, 0x5102, 0x3304, 0x1326, 0x8267, 0x1146,
    ],
    &[0x4223, 0x8267, 0x1146, 0x5102, 0x2315, 0x2245, 0x8157],
    &[
        0x3304, 0x6201, 0x4223, 0x8267, 0x1146, 0x2315, 0x2245, 0x8157,
    ],
    &[
        0x4223, 0x8267, 0x1146, 0x5102, 0x6201, 0x2245, 0x8157, 0x4113,
    ],
    &[0x3304, 0x2245, 0x8157, 0x4113, 0x4223, 0x8267, 0x1146],
    &[
        0x4223, 0x4113, 0x8337, 0x2315, 0x2245, 0x8157, 0x1326, 0x8267, 0x1146,
    ],
    &[
        0x6201, 0x5102, 0x3304, 0x4223, 0x4113, 0x8337, 0x2315, 0x2245, 0x8157, 0x1326, 0x8267,
        0x1146,
    ],
    &[
        0x8337, 0x4223, 0x6201, 0x2245, 0x8157, 0x1326, 0x8267, 0x1146,
    ],
    &[
        0x4223, 0x5102, 0x3304, 0x2245, 0x8157, 0x8337, 0x1326, 0x8267, 0x1146,
    ],
    &[
        0x8267, 0x1146, 0x5102, 0x4113, 0x8337, 0x2315, 0x2245, 0x8157,
    ],
    &[
        0x6201, 0x4113, 0x8337, 0x8267, 0x1146, 0x3304, 0x2315, 0x2245, 0x8157,
    ],
    &[0x8337, 0x8267, 0x1146, 0x5102, 0x6201, 0x2245, 0x8157],
    &[0x3304, 0x2245, 0x8157, 0x8337, 0x8267, 0x1146],
    &[0x8157, 0x2315, 0x3304, 0x1326, 0x8267],
    &[0x8267, 0x8157, 0x2315, 0x6201, 0x5102, 0x1326],
    &[0x8267, 0x1326, 0x3304, 0x6201, 0x4113, 0x8157],
    &[0x8267, 0x8157, 0x4113, 0x5102, 0x1326],
    &[0x5102, 0x4223, 0x8267, 0x8157, 0x2315, 0x3304],
    &[0x2315, 0x6201, 0x4223, 0x8267, 0x8157],
    &[0x3304, 0x5102, 0x4223, 0x8267, 0x8157, 0x4113, 0x6201],
    &[0x4113, 0x4223, 0x8267, 0x8157],
    &[
        0x8157, 0x2315, 0x3304, 0x1326, 0x8267, 0x4223, 0x4113, 0x8337,
    ],
    &[
        0x8157, 0x2315, 0x6201, 0x5102, 0x1326, 0x8267, 0x4223, 0x4113, 0x8337,
    ],
    &[0x8157, 0x8337, 0x4223, 0x6201, 0x3304, 0x1326, 0x8267],
    &[0x5102, 0x1326, 0x8267, 0x8157, 0x8337, 0x4223],
    &[0x8267, 0x8157, 0x2315, 0x3304, 0x5102, 0x4113, 0x8337],
    &[0x6201, 0x4113, 0x8337, 0x8267, 0x8157, 0x2315],
    &[0x6201, 0x3304, 0x5102, 0x8337, 0x8267, 0x8157],
    &[0x8337, 0x8267, 0x8157],
    &[0x8337, 0x8157, 0x8267],
    &[0x6201, 0x5102, 0x3304, 0x8337, 0x8157, 0x8267],
    &[0x6201, 0x2315, 0x4113, 0x8337, 0x8157, 0x8267],
    &[0x5102, 0x3304, 0x2315, 0x4113, 0x8337, 0x8157, 0x8267],
    &[0x5102, 0x4223, 0x1326, 0x8337, 0x8157, 0x8267],
    &[0x6201, 0x4223, 0x1326, 0x3304, 0x8337, 0x8157, 0x8267],
    &[
        0x6201, 0x2315, 0x4113, 0x5102, 0x4223, 0x1326, 0x8337, 0x8157, 0x8267,
    ],
    &[
        0x4223, 0x1326, 0x3304, 0x2315, 0x4113, 0x8337, 0x8157, 0x8267,
    ],
    &[0x4113, 0x8157, 0x8267, 0x4223],
    &[0x4223, 0x4113, 0x8157, 0x8267, 0x6201, 0x5102, 0x3304],
    &[0x8157, 0x8267, 0x4223, 0x6201, 0x2315],
    &[0x3304, 0x2315, 0x8157, 0x8267, 0x4223, 0x5102],
    &[0x1326, 0x5102, 0x4113, 0x8157, 0x8267],
    &[0x8157, 0x4113, 0x6201, 0x3304, 0x1326, 0x8267],
    &[0x1326, 0x5102, 0x6201, 0x2315, 0x8157, 0x8267],
    &[0x8267, 0x1326, 0x3304, 0x2315, 0x8157],
    &[0x3304, 0x1146, 0x2245, 0x8337, 0x8157, 0x8267],
    &[0x6201, 0x5102, 0x1146, 0x2245, 0x8337, 0x8157, 0x8267],
    &[
        0x6201, 0x2315, 0x4113, 0x3304, 0x1146, 0x2245, 0x8337, 0x8157, 0x8267,
    ],
    &[
        0x2315, 0x4113, 0x5102, 0x1146, 0x2245, 0x8337, 0x8157, 0x8267,
    ],
    &[
        0x5102, 0x4223, 0x1326, 0x3304, 0x1146, 0x2245, 0x8337, 0x8157, 0x8267,
    ],
    &[
        0x1146, 0x2245, 0x6201, 0x4223, 0x1326, 0x8337, 0x8157, 0x8267,
    ],
    &[
        0x6201, 0x2315, 0x4113, 0x5102, 0x4223, 0x1326, 0x3304, 0x1146, 0x2245, 0x8337, 0x8157,
        0x8267,
    ],
    &[
        0x4113, 0x4223, 0x1326, 0x1146, 0x2245, 0x2315, 0x8337, 0x8157, 0x8267,
    ],
    &[0x4223, 0x4113, 0x8157, 0x8267, 0x3304, 0x1146, 0x2245],
    &[
        0x6201, 0x5102, 0x1146, 0x2245, 0x4223, 0x4113, 0x8157, 0x8267,
    ],
    &[
        0x8157, 0x8267, 0x4223, 0x6201, 0x2315, 0x3304, 0x1146, 0x2245,
    ],
    &[0x2315, 0x8157, 0x8267, 0x4223, 0x5102, 0x1146, 0x2245],
    &[
        0x1326, 0x5102, 0x4113, 0x8157, 0x8267, 0x3304, 0x1146, 0x2245,
    ],
    &[0x1326, 0x1146, 0x2245, 0x6201, 0x4113, 0x8157, 0x8267],
    &[
        0x5102, 0x6201, 0x2315, 0x8157, 0x8267, 0x1326, 0x3304, 0x1146, 0x2245,
    ],
    &[0x1326, 0x1146, 0x2245, 0x2315, 0x8157, 0x8267],
    &[0x2315, 0x2245, 0x8267, 0x8337],
    &[0x2315, 0x2245, 0x8267, 0x8337, 0x6201, 0x5102, 0x3304],
    &[0x4113, 0x6201, 0x2245, 0x8267, 0x8337],
    &[0x5102, 0x4113, 0x8337, 0x8267, 0x2245, 0x3304],
    &[0x2315, 0x2245, 0x8267, 0x8337, 0x5102, 0x4223, 0x1326],
    &[
        0x6201, 0x4223, 0x1326, 0x3304, 0x8337, 0x2315, 0x2245, 0x8267,
    ],
    &[
        0x4113, 0x6201, 0x2245, 0x8267, 0x8337, 0x5102, 0x4223, 0x1326,
    ],
    &[0x4113, 0x4223, 0x1326, 0x3304, 0x2245, 0x8267, 0x8337],
    &[0x2315, 0x2245, 0x8267, 0x4223, 0x4113],
    &[
        0x2315, 0x2245, 0x8267, 0x4223, 0x4113, 0x6201, 0x5102, 0x3304,
    ],
    &[0x6201, 0x2245, 0x8267, 0x4223],
    &[0x3304, 0x2245, 0x8267, 0x4223, 0x5102],
    &[0x5102, 0x4113, 0x2315, 0x2245, 0x8267, 0x1326],
    &[0x4113, 0x2315, 0x2245, 0x8267, 0x1326, 0x3304, 0x6201],
    &[0x5102, 0x6201, 0x2245, 0x8267, 0x1326],
    &[0x3304, 0x2245, 0x8267, 0x1326],
    &[0x8267, 0x8337, 0x2315, 0x3304, 0x1146],
    &[0x5102, 0x1146, 0x8267, 0x8337, 0x2315, 0x6201],
    &[0x3304, 0x1146, 0x8267, 0x8337, 0x4113, 0x6201],
    &[0x8337, 0x4113, 0x5102, 0x1146, 0x8267],
    &[
        0x8267, 0x8337, 0x2315, 0x3304, 0x1146, 0x5102, 0x4223, 0x1326,
    ],
    &[0x1146, 0x8267, 0x8337, 0x2315, 0x6201, 0x4223, 0x1326],
    &[
        0x8267, 0x8337, 0x4113, 0x6201, 0x3304, 0x1146, 0x5102, 0x4223, 0x1326,
    ],
    &[0x4113, 0x4223, 0x1326, 0x1146, 0x8267, 0x8337],
    &[0x3304, 0x2315, 0x4113, 0x4223, 0x8267, 0x1146],
    &[0x2315, 0x6201, 0x5102, 0x1146, 0x8267, 0x4223, 0x4113],
    &[0x1146, 0x8267, 0x4223, 0x6201, 0x3304],
    &[0x5102, 0x1146, 0x8267, 0x4223],
    &[0x8267, 0x1326, 0x5102, 0x4113, 0x2315, 0x3304, 0x1146],
    &[0x6201, 0x4113, 0x2315, 0x1326, 0x1146, 0x8267],
    &[0x6201, 0x3304, 0x1146, 0x8267, 0x1326, 0x5102],
    &[0x1326, 0x1146, 0x8267],
    &[0x1326, 0x8337, 0x8157, 0x1146],
    &[0x8337, 0x8157, 0x1146, 0x1326, 0x6201, 0x5102, 0x3304],
    &[0x8337, 0x8157, 0x1146, 0x1326, 0x6201, 0x2315, 0x4113],
    &[
        0x4113, 0x5102, 0x3304, 0x2315, 0x1326, 0x8337, 0x8157, 0x1146,
    ],
    &[0x8337, 0x8157, 0x1146, 0x5102, 0x4223],
    &[0x6201, 0x4223, 0x8337, 0x8157, 0x1146, 0x3304],
    &[
        0x8337, 0x8157, 0x1146, 0x5102, 0x4223, 0x6201, 0x2315, 0x4113,
    ],
    &[0x4223, 0x8337, 0x8157, 0x1146, 0x3304, 0x2315, 0x4113],
    &[0x4223, 0x4113, 0x8157, 0x1146, 0x1326],
    &[
        0x4223, 0x4113, 0x8157, 0x1146, 0x1326, 0x6201, 0x5102, 0x3304,
    ],
    &[0x1146, 0x8157, 0x2315, 0x6201, 0x4223, 0x1326],
    &[0x4223, 0x5102, 0x3304, 0x2315, 0x8157, 0x1146, 0x1326],
    &[0x4113, 0x8157, 0x1146, 0x5102],
    &[0x6201, 0x4113, 0x8157, 0x1146, 0x3304],
    &[0x2315, 0x8157, 0x1146, 0x5102, 0x6201],
    &[0x2315, 0x8157, 0x1146, 0x3304],
    &[0x2245, 0x3304, 0x1326, 0x8337, 0x8157],
    &[0x6201, 0x2245, 0x8157, 0x8337, 0x1326, 0x5102],
    &[
        0x2245, 0x3304, 0x1326, 0x8337, 0x8157, 0x6201, 0x2315, 0x4113,
    ],
    &[0x2245, 0x2315, 0x4113, 0x5102, 0x1326, 0x8337, 0x8157],
    &[0x4223, 0x8337, 0x8157, 0x2245, 0x3304, 0x5102],
    &[0x8157, 0x2245, 0x6201, 0x4223, 0x8337],
    &[
        0x2245, 0x3304, 0x5102, 0x4223, 0x8337, 0x8157, 0x4113, 0x6201, 0x2315,
    ],
    &[0x4223, 0x8337, 0x8157, 0x2245, 0x2315, 0x4113],
    &[0x4113, 0x8157, 0x2245, 0x3304, 0x1326, 0x4223],
    &[0x1326, 0x4223, 0x4113, 0x8157, 0x2245, 0x6201, 0x5102],
    &[0x8157, 0x2245, 0x3304, 0x1326, 0x4223, 0x6201, 0x2315],
    &[0x5102, 0x1326, 0x4223, 0x2315, 0x8157, 0x2245],
    &[0x3304, 0x5102, 0x4113, 0x8157, 0x2245],
    &[0x4113, 0x8157, 0x2245, 0x6201],
    &[0x5102, 0x6201, 0x2315, 0x8157, 0x2245, 0x3304],
    &[0x2315, 0x8157, 0x2245],
    &[0x1146, 0x1326, 0x8337, 0x2315, 0x2245],
    &[
        0x1146, 0x1326, 0x8337, 0x2315, 0x2245, 0x6201, 0x5102, 0x3304,
    ],
    &[0x6201, 0x2245, 0x1146, 0x1326, 0x8337, 0x4113],
    &[0x2245, 0x1146, 0x1326, 0x8337, 0x4113, 0x5102, 0x3304],
    &[0x5102, 0x1146, 0x2245, 0x2315, 0x8337, 0x4223],
    &[0x1146, 0x3304, 0x6201, 0x4223, 0x8337, 0x2315, 0x2245],
    &[0x8337, 0x4113, 0x6201, 0x2245, 0x1146, 0x5102, 0x4223],
    &[0x4223, 0x8337, 0x4113, 0x3304, 0x2245, 0x1146],
    &[0x4113, 0x2315, 0x2245, 0x1146, 0x1326, 0x4223],
    &[
        0x1146, 0x1326, 0x4223, 0x4113, 0x2315, 0x2245, 0x6201, 0x5102, 0x3304,
    ],
    &[0x1326, 0x4223, 0x6201, 0x2245, 0x1146],
    &[0x4223, 0x5102, 0x3304, 0x2245, 0x1146, 0x1326],
    &[0x2245, 0x1146, 0x5102, 0x4113, 0x2315],
    &[0x4113, 0x2315, 0x2245, 0x1146, 0x3304, 0x6201],
    &[0x6201, 0x2245, 0x1146, 0x5102],
    &[0x3304, 0x2245, 0x1146],
    &[0x3304, 0x1326, 0x8337, 0x2315],
    &[0x5102, 0x1326, 0x8337, 0x2315, 0x6201],
    &[0x6201, 0x3304, 0x1326, 0x8337, 0x4113],
    &[0x5102, 0x1326, 0x8337, 0x4113],
    &[0x4223, 0x8337, 0x2315, 0x3304, 0x5102],
    &[0x6201, 0x4223, 0x8337, 0x2315],
    &[0x3304, 0x5102, 0x4223, 0x8337, 0x4113, 0x6201],
    &[0x4113, 0x4223, 0x8337],
    &[0x4113, 0x2315, 0x3304, 0x1326, 0x4223],
    &[0x1326, 0x4223, 0x4113, 0x2315, 0x6201, 0x5102],
    &[0x3304, 0x1326, 0x4223, 0x6201],
    &[0x5102, 0x1326, 0x4223],
    &[0x5102, 0x4113, 0x2315, 0x3304],
    &[0x6201, 0x4113, 0x2315],
    &[0x6201, 0x3304, 0x5102],
    &[],
];
