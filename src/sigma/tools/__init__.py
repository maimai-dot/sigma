"""Sigma standard tools — zero/low dependency connectors for agents.

All tools follow the BaseTool interface: subclass BaseTool, implement _run().
Function calling auto-discovers parameters via type hints.

Stdlib (zero deps):
  - HttpApiTool: GET/POST/PUT/DELETE with JSON auto-parse
  - FileSystemTool: read/write/list/exists, path-sandboxed
  - SQLDatabaseTool: SQLite queries with parameterized ? bindings
  - JsonTool: read/filter/get JSON files
  - CsvTool: read/filter/columns CSV files
  - TxtGrepTool: grep/regex text files with context
  - WebScrapeTool: HTML→text extract + JSON API fetch
  - DirectorySearchTool: recursive glob file search

Optional-dep tools:
  - PdfTool: extract text from PDF (needs PyPDF2)
  - ExcelTool: read Excel sheets (needs openpyxl)
  - DocxTool: extract text from .docx (needs python-docx)
"""

from sigma.tools.filesystem import FileSystemTool
from sigma.tools.http_api import HttpApiTool
from sigma.tools.sql_database import SQLDatabaseTool
from sigma.tools.file_search import CsvTool, JsonTool, TxtGrepTool
from sigma.tools.web_scrape import DirectorySearchTool, WebScrapeTool
from sigma.tools.document_tools import DocxTool, ExcelTool, PdfTool

__all__ = [
    "CsvTool",
    "DirectorySearchTool",
    "DocxTool",
    "ExcelTool",
    "FileSystemTool",
    "HttpApiTool",
    "JsonTool",
    "PdfTool",
    "SQLDatabaseTool",
    "TxtGrepTool",
    "WebScrapeTool",
]
