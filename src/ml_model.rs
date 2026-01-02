use crate::bbox::Bbox;
use anyhow::{Error, Result};
use image::DynamicImage;
use ort::Session;
use std::fmt;

pub struct OnnxModel {
    pub is_fp16: bool,
    pub model: Session,
}

pub enum MLModel {
    Onnx(OnnxModel),
}

pub trait Inference {
    fn load(model_path: &str, fp16: bool) -> Result<MLModel, Error>;
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Bbox>, Error>;
}

impl fmt::Display for MLModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MLModel::Onnx(onnx_model) => {
                write!(f, "ML Model using Onnx backend\n")?;
                write!(f, "{}", onnx_model)
            }
        }
    }
}

pub fn get_bbox(
    loaded_model: &MLModel,
    input_image: &DynamicImage,
    confidence_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Bbox>, Error> {
    let bboxes: Vec<Bbox> = match loaded_model {
        MLModel::Onnx(model) => {
            let res: Vec<Bbox> = model.forward(input_image, confidence_threshold, iou_threshold)?;
            res
        }
    };
    Ok(bboxes)
}
