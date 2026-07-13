// MetalColumnView — a non-owning, lightweight view over column data.
//
// Unlike `MetalColumn`, which owns its buffers via `Arc<Buffer>`, this view
// borrows `Buffer` references directly. It's intended for zero-copy kernel
// dispatch where a temporary, cheap-to-construct handle to a column's data
// (plus its offset/size metadata) is all that's needed.

use metal::Buffer;
use crate::buffer::DType;

#[derive(Clone, Copy)]
pub struct MetalColumnView<'a> {
    data: &'a Buffer,
    null_mask: Option<&'a Buffer>,
    dtype: DType,
    size: usize,
    offset: usize,
}

impl<'a> MetalColumnView<'a> {
    pub fn new(
        data: &'a Buffer,
        null_mask: Option<&'a Buffer>,
        dtype: DType,
        size: usize,
        offset: usize,
    ) -> Self {
        MetalColumnView { data, null_mask, dtype, size, offset }
    }

    pub fn data(&self) -> &Buffer { self.data }
    pub fn null_mask(&self) -> Option<&Buffer> { self.null_mask }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn size(&self) -> usize { self.size }
    pub fn offset(&self) -> usize { self.offset }

    /// Byte offset into `data` where this view's elements begin, for use
    /// when binding the buffer to a Metal kernel argument (e.g. via
    /// `set_buffer(index, Some(buffer), offset)`).
    pub fn data_ptr_offset(&self) -> u64 {
        (self.offset * self.dtype.size_in_bytes()) as u64
    }
}
