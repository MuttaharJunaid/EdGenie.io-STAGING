import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# CDN Domain Prefix requested by user
CDN_DOMAIN_PREFIX = os.getenv("CDN_DOMAIN_PREFIX", "https://assets.edgenie.io")


class FirebaseStorageManager:
    """
    Manages uploading storage files (PDFs and extracted visual images) to Firebase Storage
    and formatting public CDN URLs with the prefix 'https://assets.edgenie.io'.
    """

    def __init__(
        self,
        cdn_prefix: str = CDN_DOMAIN_PREFIX,
        bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        self.cdn_prefix = cdn_prefix.rstrip("/")
        self.bucket_name = bucket_name or os.getenv("FIREBASE_STORAGE_BUCKET", "edgenie-app.appspot.com")
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self._bucket = None

    def _init_firebase(self):
        """Initializes Firebase Admin SDK if credentials and firebase-admin are available."""
        if self._bucket is not None:
            return self._bucket

        try:
            import firebase_admin
            from firebase_admin import credentials, storage

            if not firebase_admin._apps:
                if self.credentials_path and os.path.exists(self.credentials_path):
                    cred = credentials.Certificate(self.credentials_path)
                    firebase_admin.initialize_app(cred, {"storageBucket": self.bucket_name})
                else:
                    firebase_admin.initialize_app(options={"storageBucket": self.bucket_name})

            self._bucket = storage.bucket()
            logger.info(f"Firebase Storage connected to bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Firebase Admin SDK not initialized ({e}). Operating in URL-generator mode.")
            self._bucket = False

        return self._bucket

    def build_cdn_url(self, relative_path: str) -> str:
        """
        Formats CDN URL using prefix 'https://assets.edgenie.io'.
        Example:
            relative_path: 'pdfs/5054/2010/jun/5054_s10_qp_41.pdf'
            -> 'https://assets.edgenie.io/pdfs/5054/2010/jun/5054_s10_qp_41.pdf'
        """
        clean_rel = relative_path.lstrip("/")
        return f"{self.cdn_prefix}/{clean_rel}"

    def upload_file(self, local_path: Path, destination_path: str) -> str:
        """
        Uploads local file to Firebase Storage under destination_path if Firebase is configured.
        Returns the formatted asset CDN URL with prefix 'https://assets.edgenie.io'.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        cdn_url = self.build_cdn_url(destination_path)

        bucket = self._init_firebase()
        if bucket and hasattr(bucket, "blob"):
            try:
                blob = bucket.blob(destination_path)
                blob.upload_from_filename(str(local_path))
                blob.make_public()
                logger.info(f"Uploaded {local_path.name} to Firebase Storage -> {cdn_url}")
            except Exception as e:
                logger.error(f"Failed to upload {local_path} to Firebase: {e}")

        return cdn_url

    def upload_pdf(self, pdf_path: Path) -> str:
        """
        Uploads a PDF file preserving its local filename / subject directory path structure.
        Returns asset CDN URL with prefix 'https://assets.edgenie.io'.
        """
        pdf_path = Path(pdf_path)
        # Construct path relative to pipeline/storage if possible
        try:
            rel_parts = pdf_path.relative_to(pdf_path.parents[3])
            dest_path = f"pdfs/{rel_parts}"
        except Exception:
            dest_path = f"pdfs/{pdf_path.name}"

        return self.upload_file(pdf_path, dest_path)

    def upload_visual(self, image_path: Path, paper_stem: str) -> str:
        """
        Uploads an extracted visual image file.
        Returns asset CDN URL with prefix 'https://assets.edgenie.io'.
        """
        image_path = Path(image_path)
        dest_path = f"visuals/{paper_stem}/{image_path.name}"
        return self.upload_file(image_path, dest_path)

    def sync_storage_dir(self, storage_dir: Path) -> Dict[str, str]:
        """
        Uploads all PDFs and visual images from local storage directory to Firebase Storage.
        Returns dict mapping local relative path to CDN URL.
        """
        storage_dir = Path(storage_dir).resolve()
        mapping = {}

        for root, _, files in os.walk(storage_dir):
            for file in files:
                if file.startswith(".") or file == "manifest.json":
                    continue
                file_path = Path(root) / file
                if file.endswith(".pdf"):
                    cdn_url = self.upload_pdf(file_path)
                    mapping[str(file_path.relative_to(storage_dir))] = cdn_url
                elif file.endswith(".png") or file.endswith(".jpg"):
                    paper_stem = file_path.parent.name
                    cdn_url = self.upload_visual(file_path, paper_stem)
                    mapping[str(file_path.relative_to(storage_dir))] = cdn_url

        return mapping
