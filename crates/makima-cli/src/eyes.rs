//! Animazione e rendering ASCII degli occhi concentrici di Makima.

use std::io::{self, Write};
use std::thread;
use std::time::Duration;

/// Frame ASCII dei famosi occhi ad anelli concentrici di Makima.
const FRAMES: &[&str] = &[
    // Frame 0: Occhi chiusi / fessura
    r#"
        .------------------.                    .------------------.
       /                    \                  /                    \
      |      ══════════      |                |      ══════════      |
       \                    /                  \                    /
        '------------------'                    '------------------'
"#,
    // Frame 1: Occhi semi-aperti, primi anelli
    r#"
        .------------------.                    .------------------.
       /      .------.      \                  /      .------.      \
      |      (   ══   )      |                |      (   ══   )      |
       \      '------'      /                  \      '------'      /
        '------------------'                    '------------------'
"#,
    // Frame 2: Occhi aperti, anelli concentrici ipnotici
    r#"
        .------------------.                    .------------------.
       /   .------------.   \                  /   .------------.   \
      |   /   .------.   \   |                |   /   .------.   \   |
      |  |   (   ()   )   |  |                |  |   (   ()   )   |  |
      |   \   '------'   /   |                |   \   '------'   /   |
       \   '------------'   /                  \   '------------'   /
        '------------------'                    '------------------'
"#,
    // Frame 3: Piena messa a fuoco con pupilla concentrica dorata
    r#"
        .------------------.                    .------------------.
       /   .------------.   \                  /   .------------.   \
      |   /   .------.   \   |                |   /   .------.   \   |
      |  |   (   ◉   )   |  |                |  |   (   ◉   )   |  |
      |   \   '------'   /   |                |   \   '------'   /   |
       \   '------------'   /                  \   '------------'   /
        '------------------'                    '------------------'
"#,
];

/// Pulisce lo schermo del terminale posizionando il cursore in alto a sinistra.
fn clear_terminal() {
    print!("\x1B[2J\x1B[H");
    let _ = io::stdout().flush();
}

/// Disegna un frame degli occhi con accenti dorati/gialli ANSI.
fn print_frame(frame: &str) {
    println!("\x1B[33;1m{}\x1B[0m", frame.trim_matches('\n'));
    let _ = io::stdout().flush();
}

/// Esegue l'animazione di apertura, chiusura e messa a fuoco dello sguardo di Makima.
pub fn play_eye_animation(cycles: usize) {
    let sequence = [0, 1, 2, 3, 3, 3, 2, 1, 0, 1, 2, 3];
    let frame_delay = Duration::from_millis(110);

    for _ in 0..cycles {
        for &frame_idx in &sequence {
            clear_terminal();
            println!("\x1B[90m[ Makima is observing... ]\x1B[0m\n");
            print_frame(FRAMES[frame_idx]);
            thread::sleep(frame_delay);
        }
    }
}

/// Mostra la versione statica a piena risoluzione degli occhi di Makima.
pub fn print_static_eyes() {
    print_frame(FRAMES[3]);
}
