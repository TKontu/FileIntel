# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'GraphExtractionResult' and 'GraphExtractor' models."""

import logging
import re
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.index.utils.string import clean_str
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.extract_graph import (
    CONTINUE_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    LOOP_PROMPT,
)

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]

logger = logging.getLogger(__name__)


@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph
    source_docs: dict[Any, Any]


class GraphExtractor:
    """Unipartite graph extractor class definition."""

    _model: ChatModel
    _join_descriptions: bool
    _tuple_delimiter_key: str
    _record_delimiter_key: str
    _entity_types_key: str
    _input_text_key: str
    _completion_delimiter_key: str
    _entity_name_key: str
    _input_descriptions_key: str
    _extraction_prompt: str
    _summarization_prompt: str
    _max_gleanings: int
    _on_error: ErrorHandlerFn

    def __init__(
        self,
        model_invoker: ChatModel,
        tuple_delimiter_key: str | None = None,
        record_delimiter_key: str | None = None,
        input_text_key: str | None = None,
        entity_types_key: str | None = None,
        completion_delimiter_key: str | None = None,
        prompt: str | None = None,
        join_descriptions=True,
        max_gleanings: int | None = None,
        on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        # TODO: streamline construction
        self._model = model_invoker
        self._join_descriptions = join_descriptions
        self._input_text_key = input_text_key or "input_text"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        self._entity_types_key = entity_types_key or "entity_types"
        self._extraction_prompt = prompt or GRAPH_EXTRACTION_PROMPT
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else graphrag_config_defaults.extract_graph.max_gleanings
        )
        self._on_error = on_error or (lambda _e, _s, _d: None)

    async def __call__(
        self, texts: list[str], prompt_variables: dict[str, Any] | None = None
    ) -> GraphExtractionResult:
        """Call method definition."""
        if prompt_variables is None:
            prompt_variables = {}
        all_records: dict[int, str] = {}
        source_doc_map: dict[int, str] = {}

        # Wire defaults into the prompt variables
        prompt_variables = {
            **prompt_variables,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or DEFAULT_COMPLETION_DELIMITER,
            self._entity_types_key: ",".join(
                prompt_variables[self._entity_types_key] or DEFAULT_ENTITY_TYPES
            ),
        }

        import time
        import logging
        main_logger = logging.getLogger(__name__)
        main_logger.info(f"MAIN EXTRACTION DEBUG: Processing {len(texts)} documents sequentially")

        for doc_index, text in enumerate(texts):
            doc_start_time = time.time()
            main_logger.info(f"MAIN EXTRACTION DEBUG: Starting document {doc_index}/{len(texts)}")
            try:
                # Invoke the entity extraction
                result = await self._process_document(text, prompt_variables)
                source_doc_map[doc_index] = text
                all_records[doc_index] = result
                doc_end_time = time.time()
                main_logger.info(f"MAIN EXTRACTION DEBUG: Document {doc_index} completed in {doc_end_time - doc_start_time:.2f} seconds")
            except Exception as e:
                main_logger.exception("error extracting graph")
                self._on_error(
                    e,
                    traceback.format_exc(),
                    {
                        "doc_index": doc_index,
                        "text": text,
                    },
                )

        results_start_time = time.time()
        main_logger.info(f"MAIN EXTRACTION DEBUG: Starting graph processing with {len(all_records)} document results")

        output = await self._process_results(
            all_records,
            prompt_variables.get(self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER),
            prompt_variables.get(self._record_delimiter_key, DEFAULT_RECORD_DELIMITER),
        )

        results_end_time = time.time()
        main_logger.info(f"MAIN EXTRACTION DEBUG: Graph processing completed in {results_end_time - results_start_time:.2f} seconds")

        return GraphExtractionResult(
            output=output,
            source_docs=source_doc_map,
        )

    async def _process_document(
        self, text: str, prompt_variables: dict[str, str]
    ) -> str:
        import time
        import logging
        logger = logging.getLogger(__name__)

        start_time = time.time()
        logger.info(f"EXTRACTION DEBUG: Starting document processing, text length: {len(text)}")

        response = await self._model.achat(
            self._extraction_prompt.format(**{
                **prompt_variables,
                self._input_text_key: text,
            }),
        )
        first_response_time = time.time()
        logger.info(f"EXTRACTION DEBUG: First LLM call completed in {first_response_time - start_time:.2f} seconds")

        results = response.output.content or ""
        logger.info(f"EXTRACTION DEBUG: First response length: {len(results)}, max_gleanings: {self._max_gleanings}")

        # if gleanings are specified, enter a loop to extract more entities
        # there are two exit criteria: (a) we hit the configured max, (b) the model says there are no more entities
        if self._max_gleanings > 0:
            logger.info(f"EXTRACTION DEBUG: Starting gleaning loop with {self._max_gleanings} max iterations")
            for i in range(self._max_gleanings):
                gleaning_start = time.time()
                logger.info(f"EXTRACTION DEBUG: Starting gleaning iteration {i}")

                response = await self._model.achat(
                    CONTINUE_PROMPT,
                    name=f"extract-continuation-{i}",
                    history=response.history,
                )
                gleaning_llm_time = time.time()
                logger.info(f"EXTRACTION DEBUG: Gleaning LLM call {i} completed in {gleaning_llm_time - gleaning_start:.2f} seconds")

                results += response.output.content or ""

                # if this is the final glean, don't bother updating the continuation flag
                if i >= self._max_gleanings - 1:
                    logger.info(f"EXTRACTION DEBUG: Reached max gleanings, breaking")
                    break

                loop_check_start = time.time()
                logger.info(f"EXTRACTION DEBUG: Starting loop check {i}")

                response = await self._model.achat(
                    LOOP_PROMPT,
                    name=f"extract-loopcheck-{i}",
                    history=response.history,
                )
                loop_check_time = time.time()
                logger.info(f"EXTRACTION DEBUG: Loop check {i} completed in {loop_check_time - loop_check_start:.2f} seconds, response: '{response.output.content}'")

                if response.output.content != "Y":
                    logger.info(f"EXTRACTION DEBUG: Loop check returned '{response.output.content}', breaking gleaning loop")
                    break

        end_time = time.time()
        logger.info(f"EXTRACTION DEBUG: Document processing completed in {end_time - start_time:.2f} seconds total")
        return results

    async def _process_results(
        self,
        results: dict[int, str],
        tuple_delimiter: str,
        record_delimiter: str,
    ) -> nx.Graph:
        """Parse the result string to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain
            - tuple_delimiter - delimiter between tuples in an output record, default is '<|>'
            - record_delimiter - delimiter between records, default is '##'
        Returns:
            - output - unipartite graph in graphML format
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        start_time = time.time()
        logger.info(f"PROCESS RESULTS DEBUG: Starting to process {len(results)} document results")

        graph = nx.Graph()
        for source_doc_id, extracted_data in results.items():
            doc_process_start = time.time()
            logger.info(f"PROCESS RESULTS DEBUG: Processing document {source_doc_id}, data length: {len(extracted_data)}")

            records = [r.strip() for r in extracted_data.split(record_delimiter)]

            for record in records:
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(tuple_delimiter)

                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = clean_str(record_attributes[1].upper())
                    entity_type = clean_str(record_attributes[2].upper())
                    entity_description = clean_str(record_attributes[3])

                    if entity_name in graph.nodes():
                        node = graph.nodes[entity_name]
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list({
                                    *_unpack_descriptions(node),
                                    entity_description,
                                })
                            )
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        node["source_id"] = ", ".join(
                            list({
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            })
                        )
                        node["type"] = (
                            entity_type if entity_type != "" else node["type"]
                        )
                    else:
                        graph.add_node(
                            entity_name,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )

                if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 5
                ):
                    # add this record as edge
                    source = clean_str(record_attributes[1].upper())
                    target = clean_str(record_attributes[2].upper())
                    edge_description = clean_str(record_attributes[3])
                    edge_source_id = clean_str(str(source_doc_id))
                    try:
                        weight = float(record_attributes[-1])
                    except ValueError:
                        weight = 1.0

                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                edge_description = "\n".join(
                                    list({
                                        *_unpack_descriptions(edge_data),
                                        edge_description,
                                    })
                                )
                            edge_source_id = ", ".join(
                                list({
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                })
                            )
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )

        end_time = time.time()
        logger.info(f"PROCESS RESULTS DEBUG: Graph processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"PROCESS RESULTS DEBUG: Final graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph


def _unpack_descriptions(data: Mapping) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data: Mapping) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")
