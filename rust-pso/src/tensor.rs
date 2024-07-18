pub struct Tensor {
    pub tensor: tch::Tensor,
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor {
            tensor: self.tensor.shallow_clone(),
        }
    }
}

impl Tensor {
    pub fn from_slice(slice: &[f32], device: &tch::Device) -> Tensor {
        Tensor {
            tensor: tch::Tensor::from_slice(slice).to_device(*device),
        }
    }

    pub fn into_vec(self) -> Result<Vec<f32>, tch::TchError> {
        self.tensor.try_into()
    }

    pub fn zeros(shape: impl torch_sys::IntList, device: &tch::Device) -> Tensor {
        Tensor {
            tensor: tch::Tensor::zeros(shape, tch::kind::FLOAT_CPU).to_device(*device),
        }
    }

    pub fn dims(&self) -> Vec<i64> {
        self.tensor.size()
    }

    pub fn narrow(&self, dim: i64, start: i64, length: i64) -> Result<Tensor, tch::TchError> {
        Ok(Tensor {
            tensor: self.tensor.f_narrow(dim, start, length)?,
        })
    }

    pub fn select(&self, dim: i64, index: i64) -> Result<Tensor, tch::TchError> {
        Ok(Tensor {
            tensor: self.tensor.f_select(dim, index)?,
        })
    }

    pub fn squeeze(&self, dim: i64) -> Result<Tensor, tch::TchError> {
        Ok(Tensor {
            tensor: self.tensor.f_squeeze_dim(dim)?,
        })
    }

    pub fn unsqueeze_dim(&self, dim: i64) -> Result<Tensor, tch::TchError> {
        Ok(Tensor {
            tensor: self.tensor.f_unsqueeze(dim)?,
        })
    }

    pub fn slice_dim(
        &self,
        dim: i64,
        start: impl Into<Option<i64>>,
        end: impl Into<Option<i64>>,
    ) -> Result<Tensor, tch::TchError> {
        Ok(Tensor {
            tensor: self.tensor.f_slice(dim, start, end, 1)?,
        })
    }

    pub fn reshape(&self, shape: impl torch_sys::IntList) -> Result<Tensor, tch::TchError> {
        Ok(Tensor {
            tensor: self.tensor.f_reshape(shape)?,
        })
    }

    pub fn cat(tensors: &[Tensor], dim: i64) -> Result<Tensor, tch::TchError> {
        let tensors: Vec<_> = tensors.iter().map(|t| &t.tensor).collect();
        Ok(Tensor {
            tensor: tch::Tensor::cat(&tensors, dim),
        })
    }

    pub fn stack(tensors: &[Tensor], dim: i64) -> Result<Tensor, tch::TchError> {
        let tensors: Vec<_> = tensors.iter().map(|t| &t.tensor).collect();
        Ok(Tensor {
            tensor: tch::Tensor::f_stack(&tensors, dim)?,
        })
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, tch::TchError> {
        Ok(Tensor {
            tensor: self.tensor.f_matmul(&other.tensor)?,
        })
    }

    pub fn transpose(&self, dim1: i64, dim2: i64) -> Tensor {
        Tensor {
            tensor: self.tensor.transpose(dim1, dim2)
        }
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}
