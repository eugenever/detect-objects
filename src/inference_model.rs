use std::fmt;

use anyhow::{Error, Result};
use image::DynamicImage;
use ort::Session;

use crate::bbox::Bbox;

pub struct OnnxModel {
    pub is_fp16: bool,
    pub model: Session,
    pub w: i32,
    pub h: i32,
    pub t: String,
}

pub enum InferenceModel {
    Onnx(OnnxModel),
}

pub trait Inference {
    fn load(
        model_path: &str,
        fp16: bool,
        w: i32,
        h: i32,
        t: String,
    ) -> Result<InferenceModel, Error>;
    async fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Bbox>, Error>;
}

impl fmt::Display for InferenceModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InferenceModel::Onnx(onnx_model) => {
                write!(f, "Inference Model using Onnx backend\n")?;
                write!(f, "{}", onnx_model)
            }
        }
    }
}

pub async fn get_bbox(
    loaded_model: &InferenceModel,
    input_image: &DynamicImage,
    confidence_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Bbox>, Error> {
    let bboxes: Vec<Bbox> = match loaded_model {
        InferenceModel::Onnx(model) => {
            let res: Vec<Bbox> = model
                .forward(input_image, confidence_threshold, iou_threshold)
                .await?;
            res
        }
    };
    Ok(bboxes)
}
