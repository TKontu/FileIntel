from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging

from ..prompt_management.composer import PromptComposer
from ..prompt_management.loader import PromptLoader
from ..llm_integration.base import LLMProvider
from ..core.exceptions import DocumentProcessingException

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Service for extracting structured metadata from document text using LLM analysis.

    Uses the first few chunks of a document to intelligently extract bibliographic
    metadata including title, authors, publication info, etc.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_composer: PromptComposer,
        max_chunks_for_extraction: int = 3,
    ):
        """
        Initialize the metadata extractor.

        Args:
            llm_provider: LLM provider for generating responses
            prompt_composer: Composer for building extraction prompts
            max_chunks_for_extraction: Maximum number of chunks to use for extraction
        """
        self.llm_provider = llm_provider
        self.prompt_composer = prompt_composer
        self.max_chunks_for_extraction = max_chunks_for_extraction
        self.task_name = "metadata_extraction"

    def extract_metadata(
        self, text_chunks: List[str], file_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from document chunks using LLM analysis.

        Args:
            text_chunks: List of text chunks from the document
            file_metadata: Existing metadata from file properties (optional)

        Returns:
            Merged metadata dictionary containing both file and LLM-extracted metadata
        """
        logger.info(f"Starting LLM metadata extraction from {len(text_chunks)} chunks")

        # Use first N chunks for extraction (typically contain title, authors, etc.)
        extraction_chunks = text_chunks[: self.max_chunks_for_extraction]
        combined_text = "\n\n".join(extraction_chunks)

        logger.debug(
            f"Using {len(extraction_chunks)} chunks ({len(combined_text)} chars) for extraction"
        )

        try:
            # Generate LLM extraction prompt
            context = {"document_text": combined_text}

            prompt = self.prompt_composer.compose(self.task_name, context)
            logger.debug("Generated metadata extraction prompt")

            # Get LLM response
            response = self.llm_provider.generate_response(prompt)
            logger.debug("Received LLM response for metadata extraction")

            # Parse JSON response
            llm_metadata = self._parse_llm_response(response.content)

            # Merge with existing file metadata
            merged_metadata = self._merge_metadata(file_metadata, llm_metadata)

            logger.info(
                f"Successfully extracted metadata with {len(merged_metadata)} fields"
            )
            return merged_metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}", exc_info=True)
            # Return original file metadata if LLM extraction fails
            return file_metadata or {}

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON metadata.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed metadata dictionary
        """
        try:
            # Try to find JSON in the response
            response_text = response_text.strip()

            # Handle case where response might have markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end > start:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end > start:
                    response_text = response_text[start:end].strip()

            # Parse JSON
            metadata = json.loads(response_text)

            # Validate structure
            if not isinstance(metadata, dict):
                raise ValueError("Response is not a JSON object")

            logger.debug(f"Successfully parsed LLM metadata: {list(metadata.keys())}")
            return metadata

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}...")
            return {}
        except Exception as e:
            logger.warning(f"Error processing LLM response: {e}")
            return {}

    def _merge_metadata(
        self, file_metadata: Optional[Dict[str, Any]], llm_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge file metadata with LLM-extracted metadata into clean, structured metadata.

        Only stores relevant bibliographic and document metadata, filtering out
        technical file properties and duplicates.

        Args:
            file_metadata: Metadata from file properties
            llm_metadata: Metadata extracted by LLM

        Returns:
            Clean, structured metadata dictionary
        """
        # Define canonical metadata schema - only these fields will be stored
        canonical_fields = {
            "title",
            "authors",
            "publication_date",
            "publisher",
            "doi",
            "source_url",
            "language",
            "document_type",
            "keywords",
            "abstract",
            "harvard_citation",
        }

        merged = {}

        # Extract useful fields from file metadata (not technical junk)
        if file_metadata:
            useful_file_metadata = self._extract_useful_file_metadata(file_metadata)
            merged.update(useful_file_metadata)

        # LLM metadata takes precedence for bibliographic fields
        if llm_metadata:
            clean_llm_metadata = {
                k: v
                for k, v in llm_metadata.items()
                if v is not None and v != "" and v != [] and k in canonical_fields
            }
            merged.update(clean_llm_metadata)

        # Only keep canonical fields in final metadata
        final_metadata = {
            k: v
            for k, v in merged.items()
            if k in canonical_fields and v is not None and v != "" and v != []
        }

        # Add processing metadata only if we have actual content
        if final_metadata:
            final_metadata["llm_extracted"] = True
            final_metadata["extraction_method"] = "llm_analysis"

        # Store raw file metadata separately for debugging (prefixed with _)
        if file_metadata:
            final_metadata["_raw_file_metadata"] = file_metadata

        logger.debug(f"Clean metadata fields: {list(final_metadata.keys())}")
        return final_metadata

    def _extract_useful_file_metadata(
        self, file_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract only useful metadata from file properties, filtering out technical junk.

        Args:
            file_metadata: Raw metadata from file properties

        Returns:
            Dictionary with only useful metadata fields
        """
        useful_metadata = {}

        # Map common PDF/file metadata fields to our canonical schema
        field_mappings = {
            # PDF fields -> canonical fields
            "Title": "title",
            "/Title": "title",
            "title": "title",
            "Subject": "abstract",
            "/Subject": "abstract",
            "Author": "authors",
            "/Author": "authors",
            "authors": "authors",
            "publisher": "publisher",
            "publication_date": "publication_date",
            "language": "language",
            "identifier": "doi",  # Sometimes ISBN/DOI in identifier field
            # Fields to ignore (technical junk)
            "Creator": None,  # Usually "Microsoft Word"
            "/Creator": None,
            "Producer": None,  # Usually "Adobe Acrobat"
            "/Producer": None,
            "CreationDate": None,  # File creation, not publication date
            "/CreationDate": None,
            "ModDate": None,
            "/ModDate": None,
        }

        for file_key, canonical_key in field_mappings.items():
            if file_key in file_metadata and canonical_key:
                value = file_metadata[file_key]
                if value and str(value).strip():
                    # Special handling for authors - convert to list
                    if canonical_key == "authors":
                        if isinstance(value, str):
                            # Split on common separators
                            authors = [
                                a.strip() for a in value.replace(";", ",").split(",")
                            ]
                            useful_metadata[canonical_key] = [a for a in authors if a]
                        elif isinstance(value, list):
                            useful_metadata[canonical_key] = [
                                str(a).strip() for a in value if str(a).strip()
                            ]
                    else:
                        useful_metadata[canonical_key] = str(value).strip()

        logger.debug(f"Extracted useful file metadata: {list(useful_metadata.keys())}")
        return useful_metadata

    @classmethod
    def create_from_settings(
        cls,
        llm_provider: LLMProvider,
        prompts_dir: Path,
        max_length: int,
        model_name: str,
        max_chunks: int = 3,
    ) -> "MetadataExtractor":
        """
        Factory method to create MetadataExtractor from application settings.

        Args:
            llm_provider: LLM provider instance
            prompts_dir: Directory containing prompt templates
            max_length: Maximum prompt length
            model_name: Name of the LLM model
            max_chunks: Maximum chunks to use for extraction

        Returns:
            Configured MetadataExtractor instance
        """
        loader = PromptLoader(prompts_dir=prompts_dir)
        composer = PromptComposer(
            loader=loader, max_length=max_length, model_name=model_name
        )

        return cls(
            llm_provider=llm_provider,
            prompt_composer=composer,
            max_chunks_for_extraction=max_chunks,
        )
