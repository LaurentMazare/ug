thread_local! {
     pub static KEEP_TMP: bool = {
        match std::env::var("UG_KEEP_TMP") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}
