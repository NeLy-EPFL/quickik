#[diplomat::bridge]
#[diplomat::abi_rename = "quickik_{0}"]
pub mod ffi {
    use std::fmt::Write;

    use diplomat_runtime::DiplomatWrite;

    /// What kind of error an `Error` is.
    pub enum ErrorKind {
        /// Reading a file (e.g. a body plan) failed.
        Io,
        /// A body plan's JSON couldn't be parsed.
        Json,
        /// An argument is invalid, e.g. a malformed kinematic tree, a prior
        /// that the solver doesn't support, or an observation of a
        /// nonexistent joint.
        InvalidInput,
    }

    /// The error of every fallible function. Use `quickik::unwrap` to throw it
    /// as a `quickik::Exception` instead.
    #[diplomat::opaque]
    pub struct Error(pub(crate) quickik_core::Error);

    impl Error {
        pub fn kind(&self) -> ErrorKind {
            match self.0 {
                quickik_core::Error::Io(_) => ErrorKind::Io,
                quickik_core::Error::Json(_) => ErrorKind::Json,
                quickik_core::Error::InvalidInput(_) => ErrorKind::InvalidInput,
            }
        }

        /// A human-readable description of the error.
        pub fn message(&self, out: &mut DiplomatWrite) {
            write!(out, "{}", self.0).unwrap();
        }
    }
}

/// Lets `?` convert core errors in functions returning `Result<_, Box<Error>>`.
impl From<quickik_core::Error> for Box<ffi::Error> {
    fn from(err: quickik_core::Error) -> Self {
        Box::new(ffi::Error(err))
    }
}

/// Shorthand for an `InvalidInput` error.
pub(crate) fn invalid_input(message: impl Into<String>) -> Box<ffi::Error> {
    quickik_core::Error::InvalidInput(message.into()).into()
}
