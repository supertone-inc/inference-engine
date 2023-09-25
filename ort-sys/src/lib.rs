include!("bindings.rs");

#[cfg(test)]
mod tests {
    #[test]
    fn cpp() {
        use execute_command::ExecuteCommand;
        use std::env;
        use std::process::Command;

        let cmake_install_prefix = env::var("OUT_DIR").unwrap();

        #[rustfmt::skip]
        if cfg!(unix) {
            Command::new("./build.sh")
        } else {
            Command::new("./build.bat")
        }
        .env("CMAKE_BUILD_DIR", format!("{cmake_install_prefix}/build"))
        .env("CMAKE_CONFIG", if cfg!(debug_assertions) { "Debug" } else { "Release" })
        .env("CMAKE_INSTALL_PREFIX", &cmake_install_prefix)
        .execute_status()
        .unwrap();
    }
}
