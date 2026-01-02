mod face;

use std::io::{Cursor, Read};

pub use face::{detect_face_bbox_yolo, index};

use actix_multipart::form::tempfile::TempFile;
use actix_multipart::form::MultipartForm;
use base64::{engine::general_purpose, Engine as _};
use image::{DynamicImage, ImageFormat, ImageReader};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::bbox::Bbox;

#[derive(Debug, MultipartForm)]
pub struct DetectFaceRequest {
    pub input: TempFile,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DetectFaceResponse {
    pub data: Vec<Bbox>,
    pub message: String,
}

impl DetectFaceResponse {
    pub fn respond(bbox: Vec<Bbox>, inf_time: f32) -> Self {
        DetectFaceResponse {
            data: bbox,
            message: format!("success, time: {}ms", inf_time),
        }
    }
}

pub fn tempfile_to_dynimg(input_tempfile: TempFile) -> actix_web::Result<DynamicImage> {
    let mut file = input_tempfile.file;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let img = ImageReader::new(Cursor::new(buffer))
        .with_guessed_format()?
        .decode()
        .unwrap();
    Ok(img)
}

pub fn dynimg_to_bytes(input_img: &DynamicImage) -> Vec<u8> {
    let mut img_bytes: Vec<u8> = Vec::new();
    input_img
        .write_to(&mut Cursor::new(&mut img_bytes), image::ImageFormat::Png)
        .unwrap();
    img_bytes
}

pub fn image_to_base64(img: &DynamicImage) -> Result<String, Box<dyn std::error::Error>> {
    let mut buffer = Cursor::new(Vec::new());
    img.write_to(&mut buffer, ImageFormat::Png)?;
    let bytes = buffer.into_inner();
    let base64_string = general_purpose::STANDARD.encode(bytes);
    Ok(base64_string)
}
