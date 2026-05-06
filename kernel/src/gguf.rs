//! Phase 2: GGUF Parser — Reads and extracts model metadata, vocabulary, and tensor mappings
//! from GGUF format files for loading into the transformer.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug)]
pub enum GgufError {
    InvalidMagic,
    UnsupportedVersion(u32),
    IncompleteData,
    InvalidValueType(u32),
    InvalidUtf8,
}

/// Parsed GGUF value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    /// Extract as u32, if possible.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Int32(v) => Some(*v as u32),
            GgufValue::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Extract as usize, if possible.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            GgufValue::Uint32(v) => Some(*v as usize),
            GgufValue::Int32(v) => Some(*v as usize),
            GgufValue::Uint64(v) => Some(*v as usize),
            GgufValue::Int64(v) => Some(*v as usize),
            _ => None,
        }
    }

    /// Extract as f32, if possible.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Extract as string reference, if possible.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Extract as array, if possible.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }
}

/// Information about a tensor stored in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
}

/// Parsed GGUF model containing metadata and tensor map.
pub struct GgufModel {
    pub version: u32,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub tensor_data_offset: usize,
}

impl GgufModel {
    /// Get a metadata value by key.
    pub fn get_meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get architecture string (e.g., "gemma").
    pub fn architecture(&self) -> Option<&str> {
        self.get_meta("general.architecture")?.as_str()
    }

    /// Get model dimension.
    pub fn dim(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.embedding_length", arch);
        self.get_meta(&key)?.as_usize()
    }

    /// Get number of layers.
    pub fn n_layers(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.block_count", arch);
        self.get_meta(&key)?.as_usize()
    }

    /// Get number of attention heads.
    pub fn n_heads(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.attention.head_count", arch);
        self.get_meta(&key)?.as_usize()
    }

    /// Get number of KV heads.
    pub fn n_kv_heads(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = alloc::format!("{}.attention.head_count_kv", arch);
        self.get_meta(&key)?.as_usize()
    }

    /// Get vocabulary size from tokenizer metadata.
    pub fn vocab_size(&self) -> Option<usize> {
        // Try the tokenizer model tokens array length
        if let Some(tokens) = self.get_meta("tokenizer.ggml.tokens") {
            if let Some(arr) = tokens.as_array() {
                return Some(arr.len());
            }
        }
        None
    }

    /// Get vocabulary tokens as strings.
    pub fn vocab_tokens(&self) -> Option<Vec<String>> {
        let tokens_val = self.get_meta("tokenizer.ggml.tokens")?;
        let arr = tokens_val.as_array()?;
        let mut result = Vec::with_capacity(arr.len());
        for v in arr {
            if let Some(s) = v.as_str() {
                result.push(String::from(s));
            }
        }
        Some(result)
    }

    /// Get vocabulary scores.
    pub fn vocab_scores(&self) -> Option<Vec<f32>> {
        let scores_val = self.get_meta("tokenizer.ggml.scores")?;
        let arr = scores_val.as_array()?;
        let mut result = Vec::with_capacity(arr.len());
        for v in arr {
            if let Some(f) = v.as_f32() {
                result.push(f);
            }
        }
        Some(result)
    }
}

/// Parser for the GGUF binary format.
pub struct GgufParser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GgufParser<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        if self.pos >= self.data.len() {
            return Err(GgufError::IncompleteData);
        }
        let val = self.data[self.pos];
        self.pos += 1;
        Ok(val)
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        if self.pos + 2 > self.data.len() {
            return Err(GgufError::IncompleteData);
        }
        let val = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(val)
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        if self.pos + 4 > self.data.len() {
            return Err(GgufError::IncompleteData);
        }
        let val = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(val)
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        if self.pos + 4 > self.data.len() {
            return Err(GgufError::IncompleteData);
        }
        let val = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(val)
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        if self.pos + 8 > self.data.len() {
            return Err(GgufError::IncompleteData);
        }
        let val = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(val)
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        if self.pos + 8 > self.data.len() {
            return Err(GgufError::IncompleteData);
        }
        let val = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(val)
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        if self.pos + len > self.data.len() {
            return Err(GgufError::IncompleteData);
        }
        let s = core::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|_| GgufError::InvalidUtf8)?;
        self.pos += len;
        Ok(String::from(s))
    }

    fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    /// Read a typed GGUF value.
    fn read_value(&mut self, val_type: u32) -> Result<GgufValue, GgufError> {
        match val_type {
            0 => Ok(GgufValue::Uint8(self.read_u8()?)),
            1 => Ok(GgufValue::Int8(self.read_i8()?)),
            2 => Ok(GgufValue::Uint16(self.read_u16()?)),
            3 => Ok(GgufValue::Int16(self.read_i16()?)),
            4 => Ok(GgufValue::Uint32(self.read_u32()?)),
            5 => Ok(GgufValue::Int32(self.read_i32()?)),
            6 => Ok(GgufValue::Float32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_bool()?)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let item_type = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len.min(1024)); // Cap initial alloc
                for _ in 0..len {
                    arr.push(self.read_value(item_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            10 => Ok(GgufValue::Uint64(self.read_u64()?)),
            11 => Ok(GgufValue::Int64(self.read_i64()?)),
            12 => Ok(GgufValue::Float64(self.read_f64()?)),
            _ => Err(GgufError::InvalidValueType(val_type)),
        }
    }

    /// Parse the GGUF file and return a GgufModel.
    pub fn parse(&mut self) -> Result<GgufModel, GgufError> {
        // Magic check
        if self.data.len() < 4 || &self.data[0..4] != b"GGUF" {
            return Err(GgufError::InvalidMagic);
        }
        self.pos = 4;

        let version = self.read_u32()?;
        if version < 2 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = self.read_u64()?;
        let metadata_kv_count = self.read_u64()?;

        // Read all metadata key-value pairs
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let key = self.read_string()?;
            let val_type = self.read_u32()?;
            let value = self.read_value(val_type)?;
            metadata.insert(key, value);
        }

        // Read tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = self.read_string()?;
            let n_dims = self.read_u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(self.read_u64()?);
            }
            let tensor_type = self.read_u32()?;
            let offset = self.read_u64()?;
            tensors.push(GgufTensorInfo {
                name,
                dimensions: dims,
                tensor_type,
                offset,
            });
        }

        // Tensor data starts after alignment (typically 32-byte aligned)
        let alignment = 32;
        let tensor_data_offset = (self.pos + alignment - 1) & !(alignment - 1);

        Ok(GgufModel {
            version,
            metadata,
            tensors,
            tensor_data_offset,
        })
    }
}
