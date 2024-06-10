use bevy::{
    math::bounding::Aabb3d,
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
    utils::HashMap,
};

use crate::{
    cell_data::{REGULAR_CELL_CLASS, REGULAR_CELL_DATA, REGULAR_VERTEX_DATA},
    VoxelSampler,
};

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
    pub(crate) position: Vec3,
    /// Level of details, represented a power of 2 scale. Level 0 is 1 meter units, Level 1
    /// is 2 meter units, etc. Can be negative for sub-meter precision.
    pub(crate) lod: i32,

    /// Current status of the chunk.
    pub(crate) status: VoxelChunkStatus,
}

impl VoxelChunk {
    /// Number of voxels in each dimension.
    const CHUNK_SIZE: usize = 16;

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

#[allow(clippy::needless_range_loop)]
pub(crate) fn create_voxel_mesh(sampler: &dyn VoxelSampler, origin: Vec3, lod: i32) -> Mesh {
    let scale = 2.0f32.powi(lod);
    let mut vertices: Vec<CellVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let sample_count = VoxelChunk::CHUNK_SIZE + 3;
    let mut reused_vertices: HashMap<UVec3, usize> =
        HashMap::with_capacity(sample_count * sample_count * 4);

    for x in 0..VoxelChunk::CHUNK_SIZE + 2 {
        for y in 0..VoxelChunk::CHUNK_SIZE + 2 {
            for z in 0..VoxelChunk::CHUNK_SIZE + 2 {
                // Compute the samples for each corner.
                let mut cube_case: usize = 0;
                let mut corner_samples: [f32; 8] = [0.0; 8];
                for i in 0..8 {
                    let dx = i & 1;
                    let dy = (i >> 1) & 1;
                    let dz = (i >> 2) & 1;
                    corner_samples[i] = sampler.distance(
                        Vec3::new(
                            (x + dx) as f32 - 1.0,
                            (y + dy) as f32 - 1.0,
                            (z + dz) as f32 - 1.0,
                        ) * scale
                            + origin,
                    );
                    if corner_samples[i] < 0.0 {
                        cube_case |= 1 << i;
                    }
                }

                let rclass = &REGULAR_CELL_DATA[REGULAR_CELL_CLASS[cube_case] as usize];
                if rclass.vertex_count == 0 {
                    continue;
                }

                // Location of the cell's minimum corner.
                let cell_origin = UVec3::new((x << 9) as u32, (y << 9) as u32, (z << 9) as u32);

                let rvtx = REGULAR_VERTEX_DATA[cube_case];
                let mut cell_indices: [usize; 16] = [0; 16];
                let mut cell_vertices: [Vec3; 16] = [Vec3::ZERO; 16];
                for (i, vdata) in rvtx.iter().enumerate() {
                    let n0 = ((vdata >> 4) & 0x0F) as usize;
                    let n1 = (vdata & 0x0F) as usize;
                    let mut u0 = cell_origin + corner_offset(n0 as u32) * 512;
                    let mut u1 = cell_origin + corner_offset(n1 as u32) * 512;
                    let mut s0 = corner_samples[n0];
                    let mut s1 = corner_samples[n1];

                    // For higher LODs, recursively subdivide until we find the exact crossing
                    // point.
                    for _ in 0..lod {
                        let um = (u0 + u1) / 2;
                        let sm = sampler.distance(
                            Vec3::new(
                                um.x as f32 / 512.0 - 1.0,
                                um.y as f32 / 512.0 - 1.0,
                                um.z as f32 / 512.0 - 1.0,
                            ) * scale
                                + origin,
                        );
                        if (sm >= 0.0 && s0 >= 0.0) || (sm <= 0.0 && s0 <= 0.0) {
                            u0 = um;
                            s0 = sm;
                        } else {
                            u1 = um;
                            s1 = sm;
                        }
                    }

                    let t = s1 / (s1 - s0);
                    let t = (512.0 * t) as u32;
                    let pos = (t * u0 + (512 - t) * u1) / 512;

                    // Try to re-use a vertex that has already been created.
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
                }

                let num_indices = rclass.vertex_indices.len();
                let num_triangles = num_indices / 3;
                for i in 0..num_triangles {
                    // Compute the vertices of the triangle.
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

    // Remove unused vertices, and compact all the indices.
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

    // Build the mesh.
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
