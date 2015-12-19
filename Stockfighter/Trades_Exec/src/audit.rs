use std;
use std::io::Write;

use ansi_term::Colour::{White, Green, Cyan, Yellow, Red};
use ansi_term::ANSIString;
use log;
use log::{LogRecord, LogLevel, LogMetadata, LogLevelFilter};
use time;


pub struct StockfighterLogger;

impl StockfighterLogger {
    fn level_label<'a>(level: LogLevel) -> ANSIString<'a> {
        match level {
            LogLevel::Error => Red.paint("ERROR"),
            LogLevel::Warn => Yellow.paint("WARNING"),
            LogLevel::Info => Cyan.paint("INFO"),
            LogLevel::Debug => Green.paint("DEBUG"),
            LogLevel::Trace => White.dimmed().paint("TRACE")
        }
    }
}

impl log::Log for StockfighterLogger {
    fn enabled(&self, _metadata: &LogMetadata) -> bool { true }

    fn log(&self, record: &LogRecord) {
        writeln!(&mut std::io::stderr(),
                 "[{}] {}—{}",
                 time::now().strftime("%Y-%m-%d %H:%M:%S.%f").unwrap(),
                 StockfighterLogger::level_label(record.level()),
                 record.args()).expect("couldn't write to stderr?!");
    }
}

pub fn initialize_logging() {
    log::set_logger(|max_log_level| {
        max_log_level.set(LogLevelFilter::Info);
        Box::new(StockfighterLogger)
    }).expect("couldn't initialize logging");
}
