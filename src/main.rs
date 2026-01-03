mod bbox;
mod constants;
mod detect;
mod inference_model;
mod services;

use actix_web::{dev::ServerHandle, middleware::Logger, web, App, HttpServer};
use clap::Parser;
use log::{error, info};

use constants::TypeOnnxModel;
use inference_model::{Inference, OnnxModel};
use services::{detect_bbox_yolo, index};

#[derive(clap::Parser)]
struct Cli {
    #[arg(long, default_value_t = 8090)]
    port: u16,
    #[arg(long, default_value_t = 1)]
    num_workers: u8,
    #[arg(
        long,
        default_value = "info",
        value_parser = clap::builder::PossibleValuesParser::new(
            ["trace", "debug", "info", "error"]
        ),
    )]
    log_level: String,
    #[arg(
        long,
        value_parser = clap::builder::PossibleValuesParser::new(
            [TypeOnnxModel::Face.as_ref(), TypeOnnxModel::Object.as_ref()]
        ),
    )]
    model: String,
}

/*

./target/release/detect_objects --num-workers 2 --log-level info --model face

curl --form input='@1.jpg' "http://localhost:8090/detect"

*/

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    env_logger::init_from_env(env_logger::Env::new().default_filter_or(&cli.log_level));

    let num_workers = cli.num_workers as usize;
    let port = cli.port;

    let onnx_model = match cli.model {
        val if val == TypeOnnxModel::Face.as_ref() => {
            OnnxModel::load("models/yolov8_face.onnx", false, 320, 320, val).unwrap()
        }
        val if val == TypeOnnxModel::Object.as_ref() => {
            OnnxModel::load("models/yolov8m.onnx", false, 640, 640, val).unwrap()
        }
        _ => {
            error!("Unsupported type model '{}'", cli.model);
            std::process::exit(1);
        }
    };

    info!("{}", onnx_model);

    let addr = format!("0.0.0.0:{}", port);
    info!("Server started at 127.0.0.1:{}", port);

    let data_om = web::Data::new(onnx_model);

    let server = HttpServer::new(move || {
        App::new()
            .route("/", web::get().to(index))
            .route("/detect", web::post().to(detect_bbox_yolo))
            .app_data(data_om.clone())
            .wrap(Logger::default())
            .wrap(actix_cors::Cors::permissive())
    })
    .client_request_timeout(std::time::Duration::from_secs(0))
    .keep_alive(None)
    .disable_signals()
    .shutdown_timeout(30)
    .bind(&addr)?
    .workers(num_workers)
    .run();

    let handle = server.handle();
    tokio::spawn(graceful_shutdown(handle));
    server.await?;

    Ok(())
}

async fn graceful_shutdown(handle: ServerHandle) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigquit = signal(SignalKind::quit()).unwrap();
        let mut sigterm = signal(SignalKind::terminate()).unwrap();
        let mut sigint = signal(SignalKind::interrupt()).unwrap();

        tokio::select! {
            _ = sigquit.recv() => info!("SIGQUIT received"),
            _ = sigterm.recv() => info!("SIGTERM received"),
            _ = sigint.recv() => info!("SIGINT received"),
        }
    }

    #[cfg(not(unix))]
    {
        use tokio::signal::windows::*;

        let mut sigbreak = ctrl_break().unwrap();
        let mut sigint = ctrl_c().unwrap();
        let mut sigquit = ctrl_close().unwrap();
        let mut sigterm = ctrl_shutdown().unwrap();

        tokio::select! {
            _ = sigbreak.recv() => info!("ctrl-break received"),
            _ = sigquit.recv() => info!("ctrl-c received"),
            _ = sigterm.recv() => info!("ctrl-close received"),
            _ = sigint.recv() => info!("ctrl-shutdown received"),
        }
    }

    info!("Server stopped");
    handle.stop(true).await;
}
