//! The crate's error type. Every fallible public function returns [`Result`];
//! the Python and C++ bindings turn it into an exception.

/// Everything that can go wrong in this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Reading a file (e.g. a body plan) failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A body plan's JSON couldn't be parsed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// An argument is invalid, e.g. a malformed kinematic tree, a prior that
    /// the solver doesn't support, or an observation of a nonexistent joint.
    #[error("{0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Shorthand for `return Err(Error::InvalidInput(format!(...)))`.
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::error::Error::InvalidInput(format!($($arg)*)))
    };
}
pub(crate) use bail;
