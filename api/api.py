from .deepwiki_db_analyzer import get_db_path, load_documents, report
from api.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE
from api.websocket_wiki import handle_websocket_chat
from api.simple_chat import chat_completions_stream
import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
import google.generativeai as genai
import asyncio
from api.config import get_model_config as get_model_configuration
import fnmatch
import pathspec

# Configure logging
from api.logging_config import setup_logging

EXCLUDED_DIRS = {
    '.git', '.svn', '.hg',  # Version control
    '__pycache__', '.pytest_cache', '.mypy_cache',  # Python caches
    'node_modules', 'bower_components',  # JavaScript
    '.venv', 'venv', 'env', 'virtualenv',  # Python virtual environments
    'build', 'dist', 'target', 'out',  # Build outputs
    '.idea', '.vscode', '.vs',  # IDE directories
    'coverage', '.coverage',  # Test coverage
}

EXCLUDED_FILES = {
    '.DS_Store', 'Thumbs.db',  # OS files
    '__init__.py',  # Python package markers
    '*.pyc', '*.pyo', '*.pyd',  # Python compiled files
    '*.so', '*.dll', '*.dylib',  # Compiled libraries
    '*.log',  # Log files
}

MAX_FILE_SIZE = 1024 * 1024  # 1MB limit


def is_binary_file(file_path: str) -> bool:
    """Check if a file is binary by reading its first few bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            # Check for null bytes which indicate binary content
            if b'\x00' in chunk:
                return True
            # Check for high ratio of non-text bytes
            text_chars = bytearray(
                {7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
            non_text = sum(1 for byte in chunk if byte not in text_chars)
            return non_text / len(chunk) > 0.3 if chunk else False
    except Exception:
        return False  # Assume text if we can't read it


def load_gitignore_patterns(repo_path):
    """Load and parse .gitignore patterns."""
    gitignore_path = os.path.join(repo_path, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', f)
            return spec
    return None


setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper function to get adalflow root path


def get_adalflow_default_root_path():
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Pydantic Models ---


class WikiPageIteration(BaseModel):
    iteration: int
    content: str
    timestamp: int
    model: Optional[str] = None
    provider: Optional[str] = None


class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    iterations: Optional[List[WikiPageIteration]] = None
    filePaths: List[str]
    importance: str  # Should ideally be Literal['high', 'medium', 'low']
    # Should ideally be Literal['architecture' | 'api' | 'configuration' | 'deployment' | 'data_model' | 'component' | 'general']
    page_type: Optional[str] = None
    relatedPages: List[str]


class ProcessedProjectEntry(BaseModel):
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str  # Renamed from type to repo_type for clarity with existing models
    submittedAt: int  # Timestamp
    language: str  # Extracted from filename


class RepoInfo(BaseModel):
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """
    Model for the wiki sections.
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None


class WikiCacheData(BaseModel):
    """
    Model for the data to be stored in the wiki cache.
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  # compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    # Wiki Analytics Information
    wiki_analytics: Optional[Dict[str, Any]] = None
    page_analytics: Optional[Dict[str, Dict[str, Any]]
                             ] = None  # Per page Analytics Information


class WikiCacheRequest(BaseModel):
    """
    Model for the request body when saving wiki cache.
    """
    repo: RepoInfo
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str
    # Wiki Analytics Information
    wiki_analytics: Optional[Dict[str, Any]] = None
    page_analytics: Optional[Dict[str, Dict[str, Any]]
                             ] = None  # Per page Analytics Information


class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(...,
                                  description="List of wiki pages to export")
    format: Literal["markdown",
                    "json"] = Field(..., description="Export format (markdown or json)")

# --- Model Configuration Models ---


class Model(BaseModel):
    """
    Model for LLM model configuration
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model configuration parameters")


class Provider(BaseModel):
    """
    Model for LLM provider configuration
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(...,
                                description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(
        False, description="Whether this provider supports custom models")


class ModelConfig(BaseModel):
    """
    Model for the entire model configuration
    """
    providers: List[Provider] = Field(...,
                                      description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")


class AuthorizationConfig(BaseModel):
    code: str = Field(..., description="Authorization code")


class DocumentInfo(BaseModel):
    """
    Model for detailed document information from the DeepWiki database.
    Matches the structure returned by deepwiki_db_analyzer.report(json_output=True).
    """
    file_path: Optional[str] = None
    chunk_id: Optional[str] = None
    repo: Optional[str] = None
    owner: Optional[str] = None
    language: Optional[str] = None
    source: Optional[str] = None
    source_id: Optional[str] = None
    content_length: int
    content_preview: str


@app.get("/api/db_info/{repo_name}", response_model=List[DocumentInfo])
async def get_db_info_endpoint(
    repo_name: str,
    max_docs: Optional[int] = Query(
        None, description="Maximum number of documents to display"),
):
    """
    Retrieves information about a DeepWiki database for a given repository.
    """
    logger.info(f"Fetching DB info for repository: {repo_name}")
    db_path = get_db_path(repo_name)
    try:
        # load_documents might be blocking, so run it in a thread
        docs = await asyncio.to_thread(load_documents, db_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Database not found for {repo_name} at {db_path}")
    except Exception as exc:
        logger.error(f"Error loading documents for {repo_name}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Generate the report in JSON format using the provided deepwiki_db_analyzer.report function
    # The report function returns a JSON string, which we need to parse back into a Python object
    # for FastAPI's response_model validation.
    json_report_str = await asyncio.to_thread(report, docs, json_output=True, max_docs=max_docs)

    # If json_report_str is None (which happens if there are no docs and json_output is True,
    # and the report function doesn't return an empty JSON array), return an empty list.
    if json_report_str is None:
        return []

    return json.loads(json_report_str)


@app.get("/lang/config")
async def get_lang_config():
    return configs["lang_config"]


@app.get("/auth/status")
async def get_auth_status():
    """
    Check if authentication is required for the wiki.
    """
    return {"auth_required": WIKI_AUTH_MODE}


@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """
    Check authorization code.
    """
    return {"success": WIKI_AUTH_CODE == request.code}


@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    Get available model providers and their models.

    This endpoint returns the configuration of available model providers and their
    respective models that can be used throughout the application.

    Returns:
        ModelConfig: A configuration object containing providers and their models
    """
    try:
        logger.info("Fetching model configurations")

        # Create providers from the config file
        providers = []
        default_provider = configs.get("default_provider", "google")

        # Add provider configuration based on config.py
        for provider_id, provider_config in configs["providers"].items():
            models = []
            if provider_id == "ollama":
                # Dynamically fetch Ollama models
                from api.ollama_patch import get_ollama_models

                ollama_models = get_ollama_models()
                for model_name in ollama_models:
                    model_config = get_model_configuration(
                        provider_id, model_name)
                    model_params = model_config.get("model_kwargs", {})
                    models.append(Model(
                        id=model_name,
                        name=model_name,
                        parameters=model_params
                    ))
            else:
                # Add models from config
                for model_id in provider_config["models"].keys():
                    model_config = get_model_configuration(
                        provider_id, model_id)
                    model_params = model_config.get("model_kwargs", {})
                    # Get a more user-friendly display name if possible
                    models.append(Model(
                        id=model_id,
                        name=model_id,
                        parameters=model_params
                    ))

            # Add provider with its models
            providers.append(
                Provider(
                    id=provider_id,
                    name=f"{provider_id.capitalize()}",
                    supportsCustomModel=provider_config.get(
                        "supportsCustomModel", False),
                    models=models
                )
            )

        # Create and return the full configuration
        config = ModelConfig(
            providers=providers,
            defaultProvider=default_provider
        )
        return config

    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        # Return some default configuration in case of error
        return ModelConfig(
            providers=[
                Provider(
                    id="google",
                    name="Google",
                    supportsCustomModel=True,
                    models=[
                        Model(id="gemini-2.5-flash", name="Gemini 2.5 Flash")
                    ]
                )
            ],
            defaultProvider="google"
        )


@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    Export wiki content as Markdown or JSON.

    Args:
        request: The export request containing wiki pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(
            f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # Generate Markdown content
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            # Generate JSON content
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # Create response with appropriate headers for file download
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/local_repo/structure")
async def get_local_repo_structure(
    path: str = Query(None, description="Path to local repository"),
    excluded_dirs: str = Query(
        None, description="Newline-separated list of directories to exclude"),
    excluded_files: str = Query(
        None, description="Newline-separated list of file patterns to exclude"),
    included_dirs: str = Query(
        None, description="Newline-separated list of directories to include"),
    included_files: str = Query(
        None, description="Newline-separated list of file patterns to include")
):
    """Return the file tree and README content for a local repository."""
    if not path:
        return JSONResponse(
            status_code=400,
            content={
                "error": "No path provided. Please provide a 'path' query parameter."}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found: {path}"}
        )

    try:
        logger.info(f"Processing local repository at: {path}")
        # Parse filter parameters
        excluded_dir_list = set([d.strip() for d in excluded_dirs.split(
            '\n') if d.strip()] if excluded_dirs else [])
        excluded_file_list = set([f.strip() for f in excluded_files.split(
            '\n') if f.strip()] if excluded_files else [])
        included_dir_list = set([d.strip() for d in included_dirs.split(
            '\n') if d.strip()] if included_dirs else [])
        included_file_list = set([f.strip() for f in included_files.split(
            '\n') if f.strip()] if included_files else [])

        # file_tree_lines = []
        file_tree_data = []
        readme_content = ""

        gitignore_spec = load_gitignore_patterns(path)
        final_excluded_dir_list = EXCLUDED_DIRS.union(set(excluded_dir_list))
        final_excluded_file_list = EXCLUDED_FILES.union(
            set(excluded_file_list))

        logger.info(f'final_excluded_dir_list: {final_excluded_dir_list}')
        logger.info(f'final_excluded_file_list: {final_excluded_file_list}')

        for root, dirs, files in os.walk(path):
            # eliminate dirs in EXCLUDED_DIRS and excluded_dir_list unless in included_dir_list
            dirs[:] = [d for d in dirs if (
                d not in final_excluded_dir_list) or d in included_dir_list]

            # Apply .gitignore patterns
            if gitignore_spec:
                dirs[:] = [
                    d for d in dirs if not gitignore_spec.match_file(os.path.relpath(os.path.join(root, d), path))]

            # eliminate files matching EXCLUDED_FILES using fnmatch.fnmatch unless in included_file_list

            files[:] = [
                f for f in files
                if not any(fnmatch.fnmatch(f, pattern) for pattern in final_excluded_file_list) or any(fnmatch.fnmatch(f, pattern) for pattern in included_file_list)
            ]

            # Apply .gitignore patterns
            if gitignore_spec:
                files[:] = [
                    f for f in files
                    if not gitignore_spec.match_file(os.path.relpath(os.path.join(root, f), path))
                ]

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                try:
                    stat = os.stat(file_path)
                    if stat.st_size > MAX_FILE_SIZE:
                        continue

                    if is_binary_file(file_path):
                        continue

                    file_tree_data.append({
                        'path': rel_path,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        # 'is_binary': is_binary_file(file_path)
                    })
                except OSError:
                    continue

                # Find README.md (case-insensitive)
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
                        readme_content = ""

        # file_tree_str = '\n'.join(sorted(file_tree_lines))
        return {
            # "file_tree": file_tree_str,
            "file_tree": file_tree_data,
            "readme": readme_content
        }
    except Exception as e:
        logger.error(f"Error processing local repository: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing local repository: {str(e)}"}
        )


def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate Markdown export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"

        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next(
                    (p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(
                        f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "Related topics: " + \
                    ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown


def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate JSON export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        JSON content as string
    """
    # Create a dictionary with metadata and pages
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # Convert to JSON string with pretty formatting
    return json.dumps(export_data, indent=2)


# Import the simplified chat implementation

# Add the chat_completions_stream endpoint to the main app
app.add_api_route("/chat/completions/stream",
                  chat_completions_stream, methods=["POST"])

# Add the WebSocket endpoint
app.add_websocket_route("/ws/chat", handle_websocket_chat)

# --- Wiki Cache Helper Functions ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)


def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str) -> str:
    """Generates the file path for a given wiki cache."""
    filename = f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)


async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str) -> Optional[WikiCacheData]:
    """Reads wiki cache data from the file system."""
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
            return None
    return None


async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """Saves wiki cache data to the file system."""
    cache_path = get_wiki_cache_path(
        data.repo.owner, data.repo.repo, data.repo.type, data.language)
    logger.info(f"Attempting to save wiki cache. Path: {cache_path}")
    try:
        payload = WikiCacheData(
            wiki_structure=data.wiki_structure,
            generated_pages=data.generated_pages,
            repo=data.repo,
            provider=data.provider,
            model=data.model,
            wiki_analytics=data.wiki_analytics,
            page_analytics=data.page_analytics
        )
        # Log size of data to be cached for debugging (avoid logging full content if large)
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(
                f"Payload prepared for caching. Size: {payload_size} bytes.")
        except Exception as ser_e:
            logger.warning(
                f"Could not serialize payload for size logging: {ser_e}")

        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(
            f"IOError saving wiki cache to {cache_path}: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False

# --- Wiki Cache API Endpoints ---


@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(...,
                           description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content")
):
    """
    Retrieves cached wiki data (structure and generated pages) for a repository.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        language = configs["lang_config"]["default"]

    logger.info(
        f"Attempting to retrieve wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language)
    if cached_data:
        return cached_data
    else:
        # Return 200 with null body if not found, as frontend expects this behavior
        # Or, raise HTTPException(status_code=404, detail="Wiki cache not found") if preferred
        logger.info(
            f"Wiki cache not found for {owner}/{repo} ({repo_type}), lang: {language}")
        return None


@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    Stores generated wiki data (structure and pages) to the server-side cache.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]

    if not supported_langs.__contains__(request_data.language):
        request_data.language = configs["lang_config"]["default"]

    logger.info(
        f"Attempting to save wiki cache for {request_data.repo.owner}/{request_data.repo.repo} ({request_data.repo.type}), lang: {request_data.language}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(
            status_code=500, detail="Failed to save wiki cache")


@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(...,
                           description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content"),
    authorization_code: Optional[str] = Query(
        None, description="Authorization code")
):
    """
    Deletes a specific wiki cache from the file system.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        raise HTTPException(
            status_code=400, detail="Language is not supported")

    if WIKI_AUTH_MODE:
        logger.info("check the authorization code")
        if WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(
                status_code=401, detail="Authorization code is invalid")

    logger.info(
        f"Attempting to delete wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Successfully deleted wiki cache: {cache_path}")
            return {"message": f"Wiki cache for {owner}/{repo} ({language}) deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting wiki cache {cache_path}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to delete wiki cache: {str(e)}")
    else:
        logger.warning(f"Wiki cache not found, cannot delete: {cache_path}")
        raise HTTPException(status_code=404, detail="Wiki cache not found")


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }


@app.get("/")
async def root():
    """Root endpoint to check if the API is running and list available endpoints dynamically."""
    # Collect routes dynamically from the FastAPI app
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # Skip docs and static routes
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            # Group endpoints by first path segment
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(
                    f"{method} {route.path}")

    # Optionally, sort endpoints for readability
    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": endpoints
    }

# --- Processed Projects Endpoint --- (New Endpoint)


@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    Lists all processed projects found in the wiki cache directory.
    Projects are identified by files named like: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json
    """
    project_entries: List[ProcessedProjectEntry] = []
    # WIKI_CACHE_DIR is already defined globally in the file

    try:
        if not os.path.exists(WIKI_CACHE_DIR):
            logger.info(
                f"Cache directory {WIKI_CACHE_DIR} not found. Returning empty list.")
            return []

        logger.info(f"Scanning for project cache files in: {WIKI_CACHE_DIR}")
        # Use asyncio.to_thread for os.listdir
        filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR)

        for filename in filenames:
            if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                file_path = os.path.join(WIKI_CACHE_DIR, filename)
                try:
                    # Use asyncio.to_thread for os.stat
                    stats = await asyncio.to_thread(os.stat, file_path)
                    parts = filename.replace("deepwiki_cache_", "").replace(
                        ".json", "").split('_')

                    # Expecting repo_type_owner_repo_language
                    # Example: deepwiki_cache_github_AsyncFuncAI_deepwiki-open_en.json
                    # parts = [github, AsyncFuncAI, deepwiki-open, en]
                    if len(parts) >= 4:
                        repo_type = parts[0]
                        owner = parts[1]
                        language = parts[-1]  # language is the last part
                        # repo can contain underscores
                        repo = "_".join(parts[2:-1])

                        project_entries.append(
                            ProcessedProjectEntry(
                                id=filename,
                                owner=owner,
                                repo=repo,
                                name=f"{owner}/{repo}",
                                repo_type=repo_type,
                                # Convert to milliseconds
                                submittedAt=int(stats.st_mtime * 1000),
                                language=language
                            )
                        )
                    else:
                        logger.warning(
                            f"Could not parse project details from filename: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue  # Skip this file on error

        # Sort by most recent first
        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"Found {len(project_entries)} processed project entries.")
        return project_entries

    except Exception as e:
        logger.error(
            f"Error listing processed projects from {WIKI_CACHE_DIR}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to list processed projects from server cache.")
