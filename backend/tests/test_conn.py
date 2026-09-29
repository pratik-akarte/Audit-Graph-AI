
import os
import socket
import traceback
from typing import Optional

from dotenv import load_dotenv

# Azure OpenAI
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)

# Azure AI Search
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient

# Azure authentication
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError

# HTTP
import requests


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(override=True)


# ============================================================
# CONFIGURATION
# ============================================================

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT"
)

AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
)

AZURE_SEARCH_ENDPOINT = os.getenv(
    "AZURE_SEARCH_ENDPOINT"
)

AZURE_SEARCH_API_KEY = os.getenv(
    "AZURE_SEARCH_API_KEY"
)

AZURE_SEARCH_INDEX_NAME = os.getenv(
    "AZURE_SEARCH_INDEX_NAME"
)


# ============================================================
# RESULT TRACKING
# ============================================================

results = []


def record(name: str, success: bool, message: str):
    """
    Record and print a test result.
    """

    status = "PASS" if success else "FAIL"

    results.append(
        {
            "name": name,
            "success": success,
            "message": message,
        }
    )

    print(f"\n[{status}] {name}")
    print(f"      {message}")


# ============================================================
# SECTION 1
# ENVIRONMENT CONFIGURATION
# ============================================================

def test_environment():

    print("\n" + "=" * 70)
    print("1. ENVIRONMENT CONFIGURATION")
    print("=" * 70)

    required = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "AZURE_OPENAI_CHAT_DEPLOYMENT": AZURE_OPENAI_CHAT_DEPLOYMENT,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
        "AZURE_SEARCH_API_KEY": AZURE_SEARCH_API_KEY,
        "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX_NAME,
    }

    for key, value in required.items():

        if value:
            if "KEY" in key:
                display = "SET"
            else:
                display = value

            record(
                key,
                True,
                display
            )

        else:

            record(
                key,
                False,
                "MISSING"
            )


# ============================================================
# SECTION 2
# DNS TEST
# ============================================================

def test_dns(hostname: str, service_name: str):

    print("\n" + "=" * 70)
    print(f"2. DNS TEST - {service_name}")
    print("=" * 70)

    if not hostname:
        record(
            service_name,
            False,
            "Endpoint missing"
        )
        return

    try:

        hostname = (
            hostname
            .replace("https://", "")
            .replace("http://", "")
            .rstrip("/")
            .split("/")[0]
        )

        print(f"Hostname: {hostname}")

        addresses = socket.getaddrinfo(
            hostname,
            443,
            type=socket.SOCK_STREAM
        )

        ips = sorted(
            set(
                address[4][0]
                for address in addresses
            )
        )

        record(
            service_name,
            True,
            f"DNS resolved to: {', '.join(ips)}"
        )

    except Exception as e:

        record(
            service_name,
            False,
            f"DNS resolution failed: {e}"
        )


# ============================================================
# SECTION 3
# HTTPS CONNECTIVITY
# ============================================================

def test_https(endpoint: str, service_name: str):

    print("\n" + "=" * 70)
    print(f"3. HTTPS CONNECTIVITY - {service_name}")
    print("=" * 70)

    if not endpoint:

        record(
            service_name,
            False,
            "Endpoint missing"
        )

        return

    try:

        response = requests.get(
            endpoint,
            timeout=10
        )

        record(
            service_name,
            True,
            f"HTTP {response.status_code}"
        )

    except requests.exceptions.ConnectionError as e:

        record(
            service_name,
            False,
            f"Connection failed: {e}"
        )

    except requests.exceptions.Timeout:

        record(
            service_name,
            False,
            "Connection timed out"
        )

    except Exception as e:

        record(
            service_name,
            False,
            str(e)
        )


# ============================================================
# SECTION 4
# AZURE CLI / DEFAULT CREDENTIAL
# ============================================================

def test_azure_auth():

    print("\n" + "=" * 70)
    print("4. AZURE AUTHENTICATION")
    print("=" * 70)

    try:

        credential = DefaultAzureCredential()

        token = credential.get_token(
            "https://management.azure.com/.default"
        )

        if token:

            record(
                "Azure authentication",
                True,
                "DefaultAzureCredential successfully obtained token"
            )

        else:

            record(
                "Azure authentication",
                False,
                "No token returned"
            )

    except Exception as e:

        record(
            "Azure authentication",
            False,
            str(e)
        )


# ============================================================
# SECTION 5
# AZURE OPENAI EMBEDDINGS
# ============================================================

def test_openai_embeddings():

    print("\n" + "=" * 70)
    print("5. AZURE OPENAI - EMBEDDINGS")
    print("=" * 70)

    try:

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION,
        )

        vector = embeddings.embed_query(
            "Azure connectivity test"
        )

        record(
            "Azure OpenAI embeddings",
            True,
            f"Embedding generated successfully. Dimensions: {len(vector)}"
        )

    except Exception as e:

        record(
            "Azure OpenAI embeddings",
            False,
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SECTION 6
# AZURE OPENAI CHAT
# ============================================================

def test_openai_chat():

    print("\n" + "=" * 70)
    print("6. AZURE OPENAI - CHAT")
    print("=" * 70)

    try:

        llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            temperature=0,
        )

        response = llm.invoke(
            "Reply with exactly: Azure Chat OK"
        )

        content = response.content

        record(
            "Azure OpenAI chat",
            True,
            f"Response: {content}"
        )

    except Exception as e:

        record(
            "Azure OpenAI chat",
            False,
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SECTION 7
# AZURE AI SEARCH - SERVICE
# ============================================================

def test_search_service():

    print("\n" + "=" * 70)
    print("7. AZURE AI SEARCH - SERVICE")
    print("=" * 70)

    try:

        credential = AzureKeyCredential(
            AZURE_SEARCH_API_KEY
        )

        index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=credential,
        )

        # Force an actual API request
        indexes = list(
            index_client.list_indexes()
        )

        index_names = [
            index.name
            for index in indexes
        ]

        record(
            "Azure AI Search service",
            True,
            f"Connected successfully. Indexes found: {index_names}"
        )

    except Exception as e:

        record(
            "Azure AI Search service",
            False,
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SECTION 8
# AZURE AI SEARCH - INDEX
# ============================================================

def test_search_index():

    print("\n" + "=" * 70)
    print("8. AZURE AI SEARCH - INDEX")
    print("=" * 70)

    try:

        credential = AzureKeyCredential(
            AZURE_SEARCH_API_KEY
        )

        index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=credential,
        )

        index = index_client.get_index(
            AZURE_SEARCH_INDEX_NAME
        )

        record(
            "Azure AI Search index",
            True,
            f"Index exists: {index.name}"
        )

    except Exception as e:

        record(
            "Azure AI Search index",
            False,
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SECTION 9
# AZURE AI SEARCH - QUERY
# ============================================================

def test_search_query():

    print("\n" + "=" * 70)
    print("9. AZURE AI SEARCH - QUERY")
    print("=" * 70)

    try:

        credential = AzureKeyCredential(
            AZURE_SEARCH_API_KEY
        )

        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=credential,
        )

        results_found = list(
            search_client.search(
                search_text="*",
                top=1,
            )
        )

        record(
            "Azure AI Search query",
            True,
            f"Query executed successfully. Returned {len(results_found)} result(s)"
        )

    except Exception as e:

        record(
            "Azure AI Search query",
            False,
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SECTION 10
# VIDEO INDEXER ENDPOINT
# ============================================================

def test_video_indexer_endpoint():

    print("\n" + "=" * 70)
    print("10. AZURE VIDEO INDEXER")
    print("=" * 70)

    print(
        "Video Indexer uses the authentication and endpoint "
        "configuration implemented by VideoIndexerService."
    )

    # We don't upload a video here.
    # This test checks whether the configured endpoint
    # can be reached.

    try:

        # Read possible endpoint variables.
        endpoint = (
            os.getenv("VIDEO_INDEXER_ENDPOINT")
            or os.getenv("AZURE_VIDEO_INDEXER_ENDPOINT")
        )

        if not endpoint:

            record(
                "Azure Video Indexer",
                False,
                "VIDEO_INDEXER_ENDPOINT not configured in .env"
            )

            return

        test_https(
            endpoint,
            "Azure Video Indexer endpoint"
        )

    except Exception as e:

        record(
            "Azure Video Indexer",
            False,
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary():

    print("\n\n")
    print("=" * 70)
    print("FINAL AZURE CONNECTIVITY SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0

    for result in results:

        status = (
            "PASS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"[{status:<4}] "
            f"{result['name']}"
        )

        if result["success"]:
            passed += 1
        else:
            failed += 1

    print("=" * 70)

    print(
        f"TOTAL: {len(results)} | "
        f"PASSED: {passed} | "
        f"FAILED: {failed}"
    )

    print("=" * 70)

    if failed == 0:

        print(
            "\nALL TESTS PASSED."
        )

    else:

        print(
            "\nONE OR MORE AZURE SERVICES FAILED."
        )

        print(
            "Review the individual FAIL messages above."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AZURE SERVICES CONNECTIVITY TEST")
    print("=" * 70)

    print(
        "\nThis test checks each Azure dependency independently."
    )

    # 1
    test_environment()

    # 2
    test_dns(
        AZURE_OPENAI_ENDPOINT,
        "Azure OpenAI DNS"
    )

    test_dns(
        AZURE_SEARCH_ENDPOINT,
        "Azure AI Search DNS"
    )

    # 3
    test_https(
        AZURE_OPENAI_ENDPOINT,
        "Azure OpenAI HTTPS"
    )

    test_https(
        AZURE_SEARCH_ENDPOINT,
        "Azure AI Search HTTPS"
    )

    # 4
    test_azure_auth()

    # 5
    test_openai_embeddings()

    # 6
    test_openai_chat()

    # 7
    test_search_service()

    # 8
    test_search_index()

    # 9
    test_search_query()

    # 10
    test_video_indexer_endpoint()

    # Final
    print_summary()

