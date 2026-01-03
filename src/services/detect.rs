use actix_multipart::form::MultipartForm;
use actix_web::http::header::ContentType;
use actix_web::{web, HttpRequest, HttpResponse};

use crate::inference_model::{get_bbox, InferenceModel};
use crate::services::tempfile_to_dynimg;
use crate::services::{DetectRequest, DetectResponse};

pub async fn index(_req: HttpRequest) -> HttpResponse {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .insert_header(("X-Hdr", "sample"))
        .body("server is up :)")
}

pub async fn detect_bbox_yolo(
    loaded_model: web::Data<InferenceModel>,
    form: MultipartForm<DetectRequest>,
    _req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let req = form.into_inner();
    let temp_file = req.input;
    let img = tempfile_to_dynimg(temp_file)?;

    let start = std::time::Instant::now();
    let bbox = get_bbox(loaded_model.get_ref(), &img, 0.7, 0.7)
        .await
        .unwrap();
    let duration = start.elapsed().as_millis() as f32;

    Ok(HttpResponse::Ok().json(DetectResponse::respond(bbox, duration)))
}
