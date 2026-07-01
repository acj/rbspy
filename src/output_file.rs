//! Helper for creating profile output files safely.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::path::Path;

/// Create (or overwrite) an output file for recorded or reported profile data.
///
/// Compared to a plain `File::create`, this:
///
/// - refuses to follow a symlink in the final path component when running as root
///   (Unix), so that a privileged rbspy can't be tricked into clobbering an arbitrary
///   file via a symlink planted at a predictable output path
/// - creates new files with mode 0o600 (Unix), since profile data can reveal the
///   internals of the profiled application
/// - when running under sudo, hands ownership of the file back to the invoking user
///   so that they can read it without root
pub fn create(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        if nix::unistd::Uid::effective().is_root() {
            options.custom_flags(libc::O_NOFOLLOW);
        }
        options.mode(0o600);
    }

    let file = options.open(path).with_context(|| {
        #[cfg(unix)]
        if path
            .symlink_metadata()
            .map(|m| m.is_symlink())
            .unwrap_or(false)
        {
            return format!(
                "Refusing to write through the symlink {} while running as root. \
                 Please give the real path, or `-` for standard output.",
                path.display()
            );
        }
        format!("Failed to create output file {}", path.display())
    })?;

    #[cfg(unix)]
    give_to_sudo_user(&file, path)?;

    Ok(file)
}

/// If we're running as root on behalf of another user via sudo, chown the freshly
/// created output file to that user so the profile is readable once sudo exits.
#[cfg(unix)]
fn give_to_sudo_user(file: &File, path: &Path) -> Result<()> {
    use std::os::fd::AsRawFd;

    if !nix::unistd::Uid::effective().is_root() {
        return Ok(());
    }
    let sudo_id = |var: &str| std::env::var(var).ok().and_then(|s| s.parse::<u32>().ok());
    let (uid, gid) = match (sudo_id("SUDO_UID"), sudo_id("SUDO_GID")) {
        (Some(uid), Some(gid)) => (uid, gid),
        _ => return Ok(()),
    };
    if unsafe { libc::fchown(file.as_raw_fd(), uid, gid) } != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "Failed to change ownership of output file {}",
                path.display()
            )
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let file = create(&path).unwrap();
        drop(file);
        assert!(path.is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = path.metadata().unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn test_overwrites_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        std::fs::write(&path, "existing contents").unwrap();
        let file = create(&path).unwrap();
        drop(file);
        assert_eq!(std::fs::read(&path).unwrap().len(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_is_rejected_when_root() {
        if !nix::unistd::Uid::effective().is_root() {
            println!("Skipping test because we're not running as root");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.txt");
        std::fs::write(&target, "precious").unwrap();
        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(create(&link).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"precious");
    }
}
