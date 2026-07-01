/// Storage formats, and io functions for rbspy's internal raw storage format.
///
/// rbspy has a versioned "raw" storage format. The versioning info is stored,
/// along with a "magic number" at the start of the file. The magic number plus
/// version are the first 8 bytes of the file, and are represented as
///
///   b"rbspyXY\n"
///
/// Here, `XY` is a decimal number in [0-99]
///
/// The use of b'\n' as a terminator effectively reserves a byte, and provides
/// flexibility to go to a different version encoding scheme if this format
/// changes _way_ too much.
extern crate anyhow;
extern crate flate2;

use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::time::SystemTime;

use crate::core::types::Header;
use crate::core::types::{StackFrame, StackTrace};

use self::flate2::Compression;

use anyhow::{format_err, Context, Error, Result};
use thiserror::Error;

/// Maximum length of a single line (header or stack trace) in a raw rbspy data file.
/// Raw files are gzipped, so without a limit a small malicious file could expand into
/// an enormous single line and exhaust memory.
const MAX_LINE_BYTES: usize = 64 * 1024 * 1024;

pub struct Store {
    encoder: flate2::write::GzEncoder<File>,
}

impl Store {
    pub fn new(out_path: &Path, sample_rate: u32) -> Result<Store, Error> {
        let file = crate::output_file::create(out_path)?;
        let mut encoder = flate2::write::GzEncoder::new(file, Compression::default());
        encoder.write_all("rbspy02\n".as_bytes())?;

        let json = serde_json::to_string(&Header {
            sample_rate: Some(sample_rate),
            rbspy_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            start_time: Some(SystemTime::now()),
        })?;
        writeln!(&mut encoder, "{}", json)?;

        Ok(Store { encoder })
    }

    pub fn write(&mut self, trace: &StackTrace) -> Result<(), Error> {
        let json = serde_json::to_string(trace)?;
        writeln!(&mut self.encoder, "{}", json)?;
        Ok(())
    }

    pub fn complete(self) -> Result<(), Error> {
        // Finish explicitly instead of relying on drop, which silently discards
        // write errors (e.g. a full disk) and would leave a truncated file behind
        // without telling anyone
        self.encoder
            .finish()
            .context("couldn't finish writing raw data file")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct Version(u64);

impl ::std::fmt::Display for Version {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Version {
    /// Parse bytes to a version.
    ///
    /// # Errors
    /// Fails with `StorageError::Invalid` if the version tag is in an unknown
    /// format.
    fn try_from(b: &[u8]) -> Result<Version, StorageError> {
        if &b[0..3] == "00\n".as_bytes() {
            Ok(Version(0))
        } else if &b[0..3] == "01\n".as_bytes() {
            Ok(Version(1))
        } else if &b[0..3] == "02\n".as_bytes() {
            Ok(Version(2))
        } else {
            Err(StorageError::Invalid)
        }
    }
}

#[derive(Error, Debug)]
pub(crate) enum StorageError {
    /// The file doesn't begin with the magic tag `rbspy` + version number.
    #[error("Invalid rbspy file")]
    Invalid,
    /// The version of the rbspy file can't be handled by this version of rbspy.
    #[error("Cannot handle rbspy format {}", _0)]
    UnknownVersion(Version),
    /// An IO error occurred.
    #[error("IO error {:?}", _0)]
    Io(io::Error),
}

fn read_version(r: &mut dyn Read) -> Result<Version, StorageError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| match e.kind() {
        // A file that's too short to contain the magic number isn't an rbspy file
        io::ErrorKind::UnexpectedEof => StorageError::Invalid,
        _ => StorageError::Io(e),
    })?;
    match &buf[..5] {
        b"rbspy" => Ok(Version::try_from(&buf[5..])?),
        _ => Err(StorageError::Invalid),
    }
}

/// Read one newline-terminated line, enforcing a maximum line length. Returns
/// `Ok(None)` at the end of the stream.
fn read_line_bounded<R: BufRead>(reader: &mut R) -> Result<Option<String>, Error> {
    let mut buf = Vec::new();
    let n = reader
        .by_ref()
        .take(MAX_LINE_BYTES as u64 + 1)
        .read_until(b'\n', &mut buf)
        .map_err(StorageError::Io)?;
    if n == 0 {
        return Ok(None);
    }
    if buf.len() > MAX_LINE_BYTES {
        return Err(format_err!(
            "line in rbspy data file exceeds the maximum length of {} bytes",
            MAX_LINE_BYTES
        ));
    }
    if buf.last() == Some(&b'\n') {
        buf.pop();
        if buf.last() == Some(&b'\r') {
            buf.pop();
        }
    }
    Ok(Some(String::from_utf8(buf).context(
        "rbspy data file contains a line that isn't valid UTF-8",
    )?))
}

/// A streaming reader for raw rbspy data files. Parses one stack trace at a time so
/// that generating a report doesn't require holding the entire profile in memory.
pub(crate) struct TraceStream<R: Read> {
    version: Version,
    #[allow(dead_code)]
    pub header: Header,
    reader: BufReader<flate2::read::GzDecoder<R>>,
}

impl<R: Read> TraceStream<R> {
    pub fn new(r: R) -> Result<TraceStream<R>, Error> {
        let mut reader = flate2::read::GzDecoder::new(r);
        let version = read_version(&mut reader)?;
        match version {
            Version(0) | Version(1) | Version(2) => (),
            v => return Err(StorageError::UnknownVersion(v).into()),
        }

        let mut reader = BufReader::new(reader);
        let header = if version == Version(2) {
            let header_line = read_line_bounded(&mut reader)?
                .ok_or_else(|| format_err!("rbspy data file is missing its header"))?;
            serde_json::from_str(&header_line).context("couldn't parse rbspy data file header")?
        } else {
            // Versions 0 and 1 don't have a header line
            Header {
                sample_rate: None,
                rbspy_version: None,
                start_time: None,
            }
        };

        Ok(TraceStream {
            version,
            header,
            reader,
        })
    }
}

impl<R: Read> Iterator for TraceStream<R> {
    type Item = Result<StackTrace, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let line = match read_line_bounded(&mut self.reader) {
                Ok(Some(line)) => line,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            };
            if line.is_empty() {
                continue;
            }
            return Some(match self.version {
                // Version 0 files contain a bare list of frames on each line
                Version(0) => serde_json::from_str::<Vec<StackFrame>>(&line)
                    .map(|trace| StackTrace {
                        trace,
                        pid: None,
                        thread_id: None,
                        time: None,
                        on_cpu: None,
                    })
                    .map_err(Error::from),
                _ => serde_json::from_str::<StackTrace>(&line).map_err(Error::from),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        let mut encoder =
            flate2::write::GzEncoder::new(Cursor::new(Vec::new()), Compression::default());
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap().into_inner()
    }

    #[test]
    fn test_reads_v2_data() {
        let data = gzip(
            concat!(
                "rbspy02\n",
                "{\"sample_rate\":100,\"rbspy_version\":\"0.0.0\",\"start_time\":null}\n",
                "{\"trace\":[{\"name\":\"f\",\"relative_path\":\"f.rb\",\"absolute_path\":null,\"lineno\":1}],\"pid\":123,\"thread_id\":null,\"time\":null,\"on_cpu\":null}\n",
            )
            .as_bytes(),
        );
        let mut stream = TraceStream::new(Cursor::new(data)).unwrap();
        assert_eq!(stream.header.sample_rate, Some(100));
        let trace = stream.next().unwrap().unwrap();
        assert_eq!(trace.pid, Some(123));
        assert_eq!(trace.trace[0].name, "f");
        assert!(stream.next().is_none());
    }

    #[test]
    fn test_empty_file_is_an_error_not_a_panic() {
        assert!(TraceStream::new(Cursor::new(Vec::new())).is_err());
    }

    #[test]
    fn test_truncated_magic_number_is_an_error_not_a_panic() {
        let data = gzip(b"rbs");
        assert!(TraceStream::new(Cursor::new(data)).is_err());
    }

    #[test]
    fn test_missing_header_is_an_error_not_a_panic() {
        let data = gzip(b"rbspy02\n");
        assert!(TraceStream::new(Cursor::new(data)).is_err());
    }

    #[test]
    fn test_garbage_input_is_an_error_not_a_panic() {
        assert!(TraceStream::new(Cursor::new(b"not even gzip".to_vec())).is_err());
        let data = gzip(b"rbspy99\nsurprise");
        assert!(TraceStream::new(Cursor::new(data)).is_err());
    }

    #[test]
    fn test_overlong_line_is_an_error_not_a_panic() {
        let mut contents = b"rbspy02\n".to_vec();
        contents.extend(vec![b'a'; MAX_LINE_BYTES + 1]);
        contents.push(b'\n');
        let data = gzip(&contents);
        assert!(TraceStream::new(Cursor::new(data)).is_err());
    }

    #[test]
    fn test_malformed_trace_line_is_an_error_not_a_panic() {
        let data = gzip(
            concat!(
                "rbspy02\n",
                "{\"sample_rate\":100,\"rbspy_version\":\"0.0.0\",\"start_time\":null}\n",
                "{\"truncated\":\n",
            )
            .as_bytes(),
        );
        let mut stream = TraceStream::new(Cursor::new(data)).unwrap();
        assert!(stream.next().unwrap().is_err());
    }
}
