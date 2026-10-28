"""Point DVC at the right remote for this environment.

DVC config files cannot read environment variables, so this writes the bucket-specific
remote into ``.dvc/config.local`` (git-ignored):

* ``S3_BUCKET`` set  -> remote ``s3`` = s3://$S3_BUCKET/dvcstore (+ AWS_ENDPOINT_URL for MinIO), made default.
* ``S3_BUCKET`` unset -> the committed default remote ``local`` (local_s3/dvcstore) is used.
"""

from __future__ import annotations

import subprocess

from src import storage
from src.utils import ROOT, get_logger

log = get_logger("configure_remote")


def dvc(*args: str, check: bool = True) -> None:
    subprocess.run(["dvc", *args], cwd=ROOT, check=check, capture_output=not check)


def main() -> None:
    if storage.is_remote():
        url = f"s3://{storage.bucket()}/dvcstore"
        dvc("remote", "add", "--local", "--force", "--default", "s3", url)
        if storage.endpoint_url():
            dvc("remote", "modify", "--local", "s3", "endpointurl", storage.endpoint_url())
        else:
            dvc("remote", "modify", "--local", "--unset", "s3", "endpointurl", check=False)
        log.info("DVC remote 's3' -> %s%s", url, f" via {storage.endpoint_url()}" if storage.endpoint_url() else "")
    else:
        dvc("remote", "remove", "--local", "s3", check=False)
        dvc("remote", "default", "--local", "--unset", check=False)
        log.info("S3_BUCKET not set: using the default DVC remote 'local' (local_s3/dvcstore)")


if __name__ == "__main__":
    main()
