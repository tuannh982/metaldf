// MetalColumn — GPU-resident column storage with ownership, offset, and
// nested children (for future variable-length / composite types such as
// Utf8, which stores its data as an offsets child + a chars child).
//
// This is Phase 1 of the storage-layer refactor: introduce the struct and
// its constructors/accessors without yet wiring it into `MetalSeries`.

use std::sync::Arc;
use metal::Buffer;
use crate::buffer::DType;
use crate::column_view::MetalColumnView;

#[derive(Clone)]
pub struct MetalColumn {
    data: Arc<Buffer>,
    null_mask: Option<Arc<Buffer>>,
    dtype: DType,
    size: usize,
    offset: usize,
    children: Vec<MetalColumn>,
}

impl MetalColumn {
    pub fn new(
        data: Arc<Buffer>,
        null_mask: Option<Arc<Buffer>>,
        dtype: DType,
        size: usize,
        offset: usize,
        children: Vec<MetalColumn>,
    ) -> Self {
        MetalColumn { data, null_mask, dtype, size, offset, children }
    }

    pub fn from_buffer(buffer: Buffer, len: usize, dtype: DType) -> Self {
        MetalColumn {
            data: Arc::new(buffer),
            null_mask: None,
            dtype,
            size: len,
            offset: 0,
            children: vec![],
        }
    }

    /// Builder-style setter attaching (or clearing) a validity bitmask.
    /// Used by `MetalSeries` constructors that build a `MetalColumn` via
    /// `from_buffer`/`new` and then know whether a `NullMask` applies.
    pub fn with_null_mask(mut self, null_mask: Option<Arc<Buffer>>) -> Self {
        self.null_mask = null_mask;
        self
    }

    pub fn data(&self) -> &Buffer { &self.data }
    pub fn data_arc(&self) -> &Arc<Buffer> { &self.data }
    pub fn null_mask(&self) -> Option<&Buffer> { self.null_mask.as_deref() }
    pub fn null_mask_arc(&self) -> Option<&Arc<Buffer>> { self.null_mask.as_ref() }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn size(&self) -> usize { self.size }
    pub fn offset(&self) -> usize { self.offset }
    pub fn num_children(&self) -> usize { self.children.len() }

    pub fn child(&self, index: usize) -> &MetalColumn {
        &self.children[index]
    }

    pub fn children(&self) -> &[MetalColumn] {
        &self.children
    }

    /// Create a lightweight, non-owning view over this column's data.
    pub fn view(&self) -> MetalColumnView<'_> {
        MetalColumnView::new(
            &self.data,
            self.null_mask.as_deref(),
            self.dtype,
            self.size,
            self.offset,
        )
    }
}
