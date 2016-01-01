use std::fs;
use std::io::Read;

use toml;

#[derive(Debug)]
pub struct Config {
    pub venue: String,
    pub symbol: String,
    pub account: String,
    pub api_key: String
}


pub fn parse_config(filename: &str) -> Config {
    let mut config_file = fs::OpenOptions::new()
        .read(true)
        .open(filename)
        .expect("couldn't open config file?!");
    let mut config_content_buffer = String::new();
    config_file.read_to_string(&mut config_content_buffer);
    let config_parser = toml::Parser::new(&config_content_buffer).parse()
        .expect("couldn't parse TOML!?");
    let config = config_parser.get("trades_exec").unwrap();
    Config {
        venue: config.lookup("venue")
            .unwrap().as_str().unwrap().to_owned(),
        symbol: config.lookup("symbol")
            .unwrap().as_str().unwrap().to_owned(),
        account: config.lookup("account")
            .unwrap().as_str().unwrap().to_owned(),
        api_key: env!("STOCKFIGHTER_API_KEY").to_owned()
    }
}


#[cfg(test)]
mod tests {
    use super::parse_config;

    #[test]
    fn concerning_config_file_parsing() {
        let config = parse_config("testex_config.toml");
        assert_eq!("EXB123456", config.account);
        assert_eq!("FOOBAR", config.symbol);
        assert_eq!("TESTEX", config.venue);
    }
}
