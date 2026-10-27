"""Exercise the optional SDK over a simulated HTTP transport, without real API calls."""

import json
import unittest

from rag.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, embed_texts

try:
    import httpx2
    from openai import AuthenticationError, OpenAI
except ImportError:
    SDK_AVAILABLE = False
else:
    SDK_AVAILABLE = True


@unittest.skipUnless(SDK_AVAILABLE, "install optional embedding dependencies in .venv")
class EmbeddingSDKTests(unittest.TestCase):
    def test_real_sdk_request_and_response_parsing_with_simulated_http(self):
        requests = []
        first = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
        second = [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)

        def handle(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/embeddings")
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "model": EMBEDDING_MODEL,
                    "data": [
                        {"object": "embedding", "index": 1, "embedding": second},
                        {"object": "embedding", "index": 0, "embedding": first},
                    ],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )

        with OpenAI(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
            max_retries=0,
        ) as client:
            result = embed_texts(["first passage", "question"], client)

        self.assertEqual(result, [first, second])
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0],
            {
                "input": ["first passage", "question"],
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIMENSIONS,
                "encoding_format": "float",
            },
        )

    def test_sdk_authentication_failure_propagates(self):
        def handle(request):
            return httpx2.Response(
                401,
                json={"error": {"message": "Test authentication failure", "type": "invalid_request_error"}},
            )

        with OpenAI(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
            max_retries=0,
        ) as client:
            with self.assertRaises(AuthenticationError):
                embed_texts(["passage"], client)
