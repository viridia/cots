use bevy::{
    math::bounding::{Aabb3d, BoundingSphere, IntersectsVolume},
    prelude::*,
};

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

pub trait VoxelSampler {
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
pub struct UnionVoxelSampler<'a> {
    pub samplers: &'a [&'a dyn VoxelSampler],
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

/// A sampler that subtractively combines the outputs of two samplers.
pub struct SubtractiveVoxelSampler<'a> {
    pub pos: &'a dyn VoxelSampler,
    pub neg: &'a dyn VoxelSampler,
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

/// A sampler that represents an infinite plane surface in 3d space.
pub struct PlaneSampler {
    normal: Vec3,
    offset: f32,
}

impl PlaneSampler {
    pub fn new(normal: Vec3, offset: f32) -> Self {
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

/// A sampler that represents a spherical volume in 3d space.
pub struct SphereSampler {
    center: Vec3,
    radius: f32,
}

impl SphereSampler {
    pub fn new(center: Vec3, radius: f32) -> Self {
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
