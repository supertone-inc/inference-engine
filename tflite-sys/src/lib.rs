include!("bindings.rs");

#[cfg(test)]
mod tests {
    #[test]
    fn cpp() {
        #[cfg(not(windows))]
        execute_command::status("./test.sh").unwrap();

        #[cfg(windows)]
        execute_command::status("./test.bat").unwrap();
    }
}
