# /// script
# dependencies = [
#     "openai>=1.0.0",
#     "python-dotenv",
#     "azure-storage-blob",
#     "azure-identity",
# ]
# ///

"""
Azure OpenAI API wrapper with Batch API and Azure Blob Storage support.
Drop-in replacement for openai_api.py that uses Azure endpoints
and supports batch processing for large-scale extraction.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, ContainerClient


# Load .env files
load_dotenv("envs/azure_batch.env")
load_dotenv("envs/azure_storage.env")

# -----------------------------
# Azure OpenAI Config
# -----------------------------
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
AZURE_API_URL = os.getenv("AZURE_API_URL", None)
AZURE_API_KEY = os.getenv("AZURE_API_KEY", None)
AZURE_ENGINE_NAME = os.getenv("AZURE_ENGINE_NAME", None)

# -----------------------------
# Azure Blob Storage Config
# -----------------------------
AZURE_STORAGE_ACCOUNT_URL = os.getenv(
    "AZURE_STORAGE_ACCOUNT_URL", "https://sroaidata.blob.core.windows.net"
)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", None)
AZURE_STORAGE_SAS_TOKEN = os.getenv("AZURE_STORAGE_SAS_TOKEN", None)
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "batch-input")


# =============================================================================
# AZURE BLOB STORAGE HELPERS
# =============================================================================


def get_blob_service_client() -> Optional[BlobServiceClient]:
    """Get Azure Blob Service Client using connection string, SAS token, or DefaultAzureCredential."""
    if AZURE_STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )

    if AZURE_STORAGE_SAS_TOKEN:
        sas = AZURE_STORAGE_SAS_TOKEN.lstrip("?")
        return BlobServiceClient(
            account_url=f"{AZURE_STORAGE_ACCOUNT_URL}?{sas}"
        )

    try:
        from azure.identity import DefaultAzureCredential

        return BlobServiceClient(
            account_url=AZURE_STORAGE_ACCOUNT_URL,
            credential=DefaultAzureCredential(),
        )
    except ImportError:
        print("azure-identity not installed, skipping DefaultAzureCredential")
    except Exception as e:
        print(f"Failed to use DefaultAzureCredential: {e}")

    return None


def get_container_client(
    container_name: Optional[str] = None,
) -> Optional[ContainerClient]:
    """Get Azure Container Client, creating the container if needed."""
    blob_service = get_blob_service_client()
    if not blob_service:
        return None
    container = container_name or AZURE_STORAGE_CONTAINER_NAME
    client = blob_service.get_container_client(container)
    try:
        client.create_container()
        print(f"Created container: {container}")
    except Exception:
        pass
    return client


def upload_file_to_blob(
    container_name: str, blob_name: str, file_path: Path
) -> None:
    """Upload a local file to Azure Blob Storage."""
    client = get_container_client(container_name)
    if not client:
        raise RuntimeError("Azure Blob Storage not configured")
    blob = client.get_blob_client(blob_name)
    with open(file_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)
    print(f"Uploaded {file_path.name} -> {container_name}/{blob_name}")


def upload_string_to_blob(
    container_name: str, blob_name: str, content: str
) -> None:
    """Upload string content to Azure Blob Storage."""
    client = get_container_client(container_name)
    if not client:
        raise RuntimeError("Azure Blob Storage not configured")
    blob = client.get_blob_client(blob_name)
    blob.upload_blob(content.encode("utf-8"), overwrite=True)
    print(f"Uploaded string -> {container_name}/{blob_name}")


def download_blob_to_file(
    container_name: str, blob_name: str, file_path: Path
) -> None:
    """Download blob content to a local file."""
    client = get_container_client(container_name)
    if not client:
        raise RuntimeError("Azure Blob Storage not configured")
    blob = client.get_blob_client(blob_name)
    with open(file_path, "wb") as f:
        f.write(blob.download_blob().readall())
    print(f"Downloaded {container_name}/{blob_name} -> {file_path}")


def download_blob_to_string(container_name: str, blob_name: str) -> str:
    """Download blob content as a string."""
    client = get_container_client(container_name)
    if not client:
        raise RuntimeError("Azure Blob Storage not configured")
    blob = client.get_blob_client(blob_name)
    return blob.download_blob().readall().decode("utf-8")


def get_blob_url_with_sas(container_name: str, blob_name: str) -> str:
    """Get the full blob URL, including SAS token if available."""
    base = f"{AZURE_STORAGE_ACCOUNT_URL}/{container_name}/{blob_name}"
    if AZURE_STORAGE_SAS_TOKEN:
        sas = AZURE_STORAGE_SAS_TOKEN.lstrip("?")
        return f"{base}?{sas}"
    return base


def download_batch_results_from_blob(
    container_client: ContainerClient,
    batch_id: str,
    output_folder_prefix: str,
    local_file: Path,
) -> bool:
    """
    Download batch results from blob storage, trying multiple path patterns.
    Returns True on success.
    """
    folder_id = batch_id.replace("batch_", "").replace("-", "")
    candidates = [
        f"{output_folder_prefix}{folder_id}/results.jsonl",
        f"{output_folder_prefix}{batch_id}/results.jsonl",
        f"{output_folder_prefix}{folder_id}.jsonl",
    ]

    for path in candidates:
        try:
            print(f"  Trying: {path}")
            blob = container_client.get_blob_client(path)
            with open(local_file, "wb") as f:
                f.write(blob.download_blob().readall())
            print(f"  Downloaded results -> {local_file}")
            return True
        except Exception:
            continue

    # Fallback: search the output folder
    print(f"  Searching {output_folder_prefix}...")
    try:
        blobs = list(
            container_client.list_blobs(name_starts_with=output_folder_prefix)
        )
        for b in blobs:
            if folder_id in b.name and b.name.endswith(".jsonl"):
                blob = container_client.get_blob_client(b.name)
                with open(local_file, "wb") as f:
                    f.write(blob.download_blob().readall())
                print(f"  Downloaded {b.name} -> {local_file}")
                return True
        for b in blobs:
            if b.name.endswith("results.jsonl") or b.name.endswith(".jsonl"):
                blob = container_client.get_blob_client(b.name)
                with open(local_file, "wb") as f:
                    f.write(blob.download_blob().readall())
                print(f"  Downloaded {b.name} -> {local_file}")
                return True
    except Exception as e:
        print(f"  Error listing blobs: {e}")

    return False


# =============================================================================
# AZURE OPENAI CLIENT
# =============================================================================


class AzureOpenAIApi:
    """
    Azure OpenAI API client supporting both synchronous and batch processing.
    Uses AzureOpenAI SDK and Azure Blob Storage for batch I/O.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.1,
        max_tokens: int = 10000,
    ):
        # Always prefer AZURE_ENGINE_NAME from env; fall back to explicit model_id
        self.model_id = AZURE_ENGINE_NAME or model_id or "gpt-4.1"
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = self._build_client(self.model_id)

    @staticmethod
    def _build_client(deployment_name: str) -> AzureOpenAI:
        if not AZURE_API_KEY:
            raise RuntimeError(
                "AZURE_API_KEY not set. "
                "Create envs/azure_batch.env with AZURE_API_KEY=<key>"
            )
        return AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_API_URL,
            azure_deployment=deployment_name,
        )

    # ── synchronous (single request) ──────────────────────────────────────

    def request(
        self,
        user_prompt: str,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Single synchronous chat completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def request_json(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> dict:
        """Synchronous request that returns parsed JSON."""
        raw = self.request(user_prompt, json_mode=True, system_prompt=system_prompt)
        return json.loads(raw)

    def request_with_retry(
        self,
        user_prompt: str,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """Request with automatic retries."""
        for attempt in range(1, max_retries + 1):
            try:
                return self.request(user_prompt, json_mode=json_mode, system_prompt=system_prompt)
            except Exception as e:
                print(f"API request failed (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    return ""
        return ""

    # ── batch helpers ─────────────────────────────────────────────────────

    @staticmethod
    def build_batch_line(
        custom_id: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        top_p: float = 0.1,
        max_tokens: int = 10000,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build one line for the batch JSONL file."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        body: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def write_batch_jsonl(
        self,
        requests: List[Dict[str, Any]],
        output_path: Path,
    ) -> List[str]:
        """
        Write a list of batch request dicts to a JSONL file.
        Returns the custom_ids.
        """
        cids: List[str] = []
        with output_path.open("w", encoding="utf-8") as f:
            for req in requests:
                cids.append(req["custom_id"])
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        print(f"Wrote {len(cids)} batch requests -> {output_path}")
        return cids

    def submit_batch(
        self,
        jsonl_path: Path,
        use_blob: bool = True,
        output_folder_prefix: str = "batch-results/",
    ) -> str:
        """
        Upload the JSONL file and submit an Azure Batch job.
        Returns the batch job ID.
        """
        batch_input_blob_url = None
        uploaded_file = None
        extra_body: Dict[str, Any] = {}

        if use_blob and get_blob_service_client():
            blob_name = f"batch-requests/{jsonl_path.name}"
            upload_file_to_blob(AZURE_STORAGE_CONTAINER_NAME, blob_name, jsonl_path)
            batch_input_blob_url = get_blob_url_with_sas(
                AZURE_STORAGE_CONTAINER_NAME, blob_name
            )
            extra_body["input_blob"] = batch_input_blob_url.split("?")[0]
            extra_body["output_folder"] = {
                "url": get_blob_url_with_sas(
                    AZURE_STORAGE_CONTAINER_NAME, output_folder_prefix
                )
            }
            print(f"Batch input blob: {batch_input_blob_url}")
        else:
            with jsonl_path.open("rb") as f:
                uploaded_file = self.client.files.create(
                    file=f,
                    purpose="batch",
                    extra_body={
                        "expires_after": {"seconds": 2592000, "anchor": "created_at"}
                    },
                )
            print(f"Uploaded file_id={uploaded_file.id}")

        batch_job = self.client.batches.create(
            input_file_id=uploaded_file.id if uploaded_file else None,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            extra_body=extra_body if extra_body else None,
        )
        print(f"Batch created: id={batch_job.id}, status={batch_job.status}")
        return batch_job.id

    def poll_batch(
        self,
        batch_id: str,
        initial_sleep: float = 5.0,
        max_sleep: float = 60.0,
    ) -> Any:
        """Poll until the batch job reaches a terminal state."""
        TERMINAL = {"completed", "failed", "cancelled", "expired"}
        sleep_time = initial_sleep
        while True:
            job = self.client.batches.retrieve(batch_id)
            counts = getattr(job, "request_counts", None)
            print(
                f"[{time.strftime('%X')}] status={job.status} counts={counts}"
            )
            if job.status in TERMINAL:
                if job.status != "completed":
                    raise RuntimeError(f"Batch ended with status={job.status}")
                return job
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 1.5, max_sleep)

    def download_batch_results(
        self,
        batch_id: str,
        job: Any,
        results_path: Path,
        use_blob: bool = True,
        output_folder_prefix: str = "batch-results/",
    ) -> Dict[str, Any]:
        """
        Download batch results and return {custom_id: result_dict}.
        """
        if use_blob and get_blob_service_client():
            container_client = get_container_client(AZURE_STORAGE_CONTAINER_NAME)
            ok = download_batch_results_from_blob(
                container_client, batch_id, output_folder_prefix, results_path
            )
            if not ok:
                raise RuntimeError(
                    f"Failed to download results from blob for batch {batch_id}"
                )
        else:
            fid = job.output_file_id
            if not fid:
                raise RuntimeError("No output_file_id on completed batch job")
            content = self.client.files.content(fid).content
            results_path.write_bytes(content)
            print(f"Saved results -> {results_path}")

        id_to_result: Dict[str, Any] = {}
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    res = json.loads(line)
                    id_to_result[res["custom_id"]] = res
        return id_to_result

    def run_batch(
        self,
        requests: List[Dict[str, Any]],
        output_dir: Path,
        prefix: str = "extraction",
        use_blob: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end batch execution:
        1. Write JSONL
        2. Submit batch
        3. Poll until done
        4. Download results
        Returns {custom_id: result_dict}.
        """
        jsonl_path = output_dir / f"{prefix}_requests.jsonl"
        results_path = output_dir / f"{prefix}_results.jsonl"

        self.write_batch_jsonl(requests, jsonl_path)
        batch_id = self.submit_batch(jsonl_path, use_blob=use_blob)
        job = self.poll_batch(batch_id)
        return self.download_batch_results(
            batch_id, job, results_path, use_blob=use_blob
        )

    # ── submit-only / resume workflow ─────────────────────────────────────

    def submit_only(
        self,
        requests: List[Dict[str, Any]],
        output_dir: Path,
        prefix: str = "extraction",
        use_blob: bool = True,
        extra_manifest: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a batch job and save a manifest file so it can be resumed later.

        The manifest JSON is written to ``<output_dir>/<prefix>_manifest.json``
        and contains everything needed to collect results later:
        - batch_id, prefix, use_blob, custom_ids, timestamp

        Returns the batch_id.
        """
        jsonl_path = output_dir / f"{prefix}_requests.jsonl"
        cids = self.write_batch_jsonl(requests, jsonl_path)
        batch_id = self.submit_batch(jsonl_path, use_blob=use_blob)

        manifest = {
            "batch_id": batch_id,
            "prefix": prefix,
            "use_blob": use_blob,
            "custom_ids": cids,
            "output_dir": str(output_dir),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra_manifest:
            manifest.update(extra_manifest)

        manifest_path = output_dir / f"{prefix}_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Manifest saved -> {manifest_path}")
        print(
            f"Job submitted.  batch_id = {batch_id}\n"
            f"Come back later and run with:\n"
            f"  --resume {batch_id}"
        )
        return batch_id

    def resume_batch(
        self,
        batch_id: str,
        results_path: Path,
        use_blob: bool = True,
    ) -> Dict[str, Any]:
        """
        Resume a previously submitted batch job by its batch_id.

        Polls until done, downloads results.
        Returns {custom_id: result_dict}.
        """
        print(f"Resuming batch {batch_id} ...")

        # Check current status first
        job = self.client.batches.retrieve(batch_id)
        counts = getattr(job, "request_counts", None)
        print(f"Current status: {job.status}  counts={counts}")

        TERMINAL = {"completed", "failed", "cancelled", "expired"}
        if job.status not in TERMINAL:
            job = self.poll_batch(batch_id)

        if job.status != "completed":
            raise RuntimeError(f"Batch ended with status={job.status}")

        return self.download_batch_results(
            batch_id, job, results_path, use_blob=use_blob
        )

    @staticmethod
    def parse_batch_response(result: Dict[str, Any]) -> Optional[str]:
        """
        Extract the text content from a single batch result entry.
        Handles both standard chat completion and reasoning-model formats.
        Returns the content string, or None on error.
        """
        if not result:
            return None
        try:
            response = result.get("response", {})
            if response.get("status_code", 0) != 200:
                return None
            body = response.get("body", {})

            # Reasoning model format (o1, o3, etc.)
            if "output" in body:
                items = body["output"]
                if items:
                    content_items = items[0].get("content", [])
                    if content_items:
                        return content_items[0].get("text", "").strip()
                return None

            # Standard chat completion format
            if "choices" in body:
                return body["choices"][0]["message"]["content"].strip()

            return None
        except (KeyError, IndexError, AttributeError, TypeError):
            return None
