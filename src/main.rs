use tokio::net::TcpListener;

mod controllers;
mod models;
mod services;

#[tokio::main]
async fn main() {
    let app = controllers::router();

    let listener = TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("无法绑定到 127.0.0.1:3000");

    println!("API server listening on http://127.0.0.1:3000");
    axum::serve(listener, app).await.expect("API server failed");
}
