//! Rendering ASCII e ritratto di Makima.

use std::io::{self, Write};
use std::thread;
use std::time::Duration;

/// Ritratto ASCII dettagliato di Makima.
pub const MAKIMA_PORTRAIT: &str = r#"
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣶⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣤⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢀⣿⣿⣿⢻⣿⣿⣿⡿⣿⣿⣟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣼⣿⣿⠇⢸⣿⣿⣿⠡⠿⣿⣿⣏⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢿⣿⣿⣀⣸⣿⣿⣿⠀⠀⣬⢿⣿⣧⡝⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠘⣿⣿⡏⠁⠻⠿⠻⠆⠈⡵⠛⢹⣿⣿⡟⠃⠁⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠹⢿⣿⡿⣿⣦⠀⠀⠀⠀⢠⠨⠿⠿⠋⠀⠀⣿⣿⣿⣿⣿⡿⠡⡪⡙⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣁⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⢿⢻⠁⢸⠎⢨⣿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣯⠋⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⢸⠀⢰⣟⡤⠉⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣏⠳⠄⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⡜⠀⣀⣀⣠⣾⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⡀⠦⠄⠠⠤⠖⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⡇⠀⢿⣿⣿⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⡇⠀⠀⣾⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⠈⡇⠀⠀⠀⠀⠀⠀⠀⠠⠔⠊⢹⣿⣿⡇⠀⠀⡏⠙⢻⣯⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⠀⠘⠦⠤⠤⣤⡀⠀⠀⠀⠀⠀⢸⣿⣿⡇⠀⠀⢰⠀⢸⠁⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⢀⡴⠚⠋⠉⡽⠹⡄⠀⠀⠀⠀⣸⣿⣿⠁⠀⠀⠈⡟⠁⠀⠸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⠏⠀⡀⠀⡸⠁⠀⡟⡄⠀⠀⠀⣿⣿⣿⠀⠀⡠⠊⠀⠀⠀⢀⣇⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⡀⢠⠃⢸⠃⠀⠀⢇⡁⠀⠀⠀⣿⣿⣿⠒⠉⠀⠀⠀⠀⠀⡌⠉⠉⠦⢄⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⡃⣏⡆⡌⠀⢀⢀⢸⠒⠒⠋⠉⣿⣿⣿⠀⠀⠀⠀⠀⠀⡜⠀⠈⠉⠳⡄⠙⠳⢦⣀⠀⠀⠀
⠀⠀⠀⠀⢹⣿⡟⠁⡆⡇⣴⣿⣿⣿⣧⡀⠀⠀⣿⣿⣿⠀⠀⠀⢀⣤⠎⠀⠀⠀⠀⠀⠸⣄⣀⡀⠉⢢⡀⠀
⠀⠀⠀⠀⣰⠿⠿⣄⠱⡋⣿⣿⣿⣿⣿⣿⠀⠀⣿⣿⣇⡠⠔⢊⡡⠊⠀⠔⢢⠀⠀⠀⠀⠀⢀⠀⠀⠀⢱⡄
⠀⠀⢠⡾⠁⠀⠀⠁⢀⣾⣿⣿⣿⣿⡟⠁⠳⠔⣻⡟⠠⢴⠰⠁⠀⢀⡔⠀⠀⠀⢀⠄⠂⡩⠔⠊⠁⠈⠁⢣
⢠⡴⠋⠀⠀⠀⠀⣴⣿⣿⣿⡟⠈⠉⠀⢀⣀⣠⡿⡁⠀⠀⠀⠀⠐⠉⢀⣀⠀⡔⠁⠰⠋⠀⠀⠀⠀⠀⠀⢸
"#;

/// Pulisce lo schermo del terminale posizionando il cursore in alto a sinistra.
fn clear_terminal() {
    print!("\x1B[2J\x1B[H");
    let _ = io::stdout().flush();
}

/// Disegna il ritratto di Makima con accenti eleganti ANSI.
fn print_portrait_styled(text: &str, color_code: &str) {
    println!("\x1B[{}m{}\x1B[0m", color_code, text.trim_matches('\n'));
    let _ = io::stdout().flush();
}

/// Esegue l'animazione di messa a fuoco e rivelazione del ritratto di Makima.
pub fn play_eye_animation(_cycles: usize) {
    let colors = ["90", "31", "33;1", "31;1", "37;1"];
    let frame_delay = Duration::from_millis(180);

    for &c in &colors {
        clear_terminal();
        println!("\x1B[90m[ Makima is observing... ]\x1B[0m\n");
        print_portrait_styled(MAKIMA_PORTRAIT, c);
        thread::sleep(frame_delay);
    }
}

/// Mostra la versione statica del ritratto di Makima.
pub fn print_static_eyes() {
    print_portrait_styled(MAKIMA_PORTRAIT, "31;1");
}
