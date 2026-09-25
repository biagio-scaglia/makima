//! Rendering ASCII e ritratto di Makima.

use std::io::{self, Write};

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

/// Disegna il ritratto di Makima con accenti eleganti ANSI.
fn print_portrait_styled(text: &str, color_code: &str) {
    println!("\x1B[{}m{}\x1B[0m", color_code, text.trim_matches('\n'));
    let _ = io::stdout().flush();
}

/// Mostra il singolo frame del ritratto di Makima (senza frame multipli duplicati).
pub fn play_eye_animation(_cycles: usize) {
    print_static_eyes();
}

/// Mostra la versione a frame singolo del ritratto di Makima.
pub fn print_static_eyes() {
    println!("\x1B[90m[ Makima is observing... ]\x1B[0m");
    print_portrait_styled(MAKIMA_PORTRAIT, "31;1");
}
