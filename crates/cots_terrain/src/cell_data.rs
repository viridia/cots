// Tables taken from https://github.com/EricLengyel/Transvoxel/blob/main/Transvoxel.cpp

pub(crate) struct RegularCellData {
    pub(crate) vertex_count: u8,
    pub(crate) vertex_indices: &'static [u8],
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

pub(crate) const REGULAR_CELL_CLASS: [u8; 256] = [
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

pub(crate) const REGULAR_CELL_DATA: [RegularCellData; 16] = [
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

pub(crate) const REGULAR_VERTEX_DATA: [&[u16]; 256] = [
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