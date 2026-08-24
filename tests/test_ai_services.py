import unittest

import requests

from portfolio_app.config import Settings
from portfolio_app.services import ExternalServiceError, OpenAIClient, PerplexityClient


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []
        self.headers = {}

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.payloads.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response if isinstance(response, FakeResponse) else FakeResponse(response)


def perplexity_payload(text: str, finish_reason: str):
    return {
        "choices": [
            {"message": {"content": text}, "finish_reason": finish_reason}
        ]
    }


def openai_payload(text: str, status: str = "completed", reason=None):
    payload = {
        "status": status,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }
    if reason:
        payload["incomplete_details"] = {"reason": reason}
    return payload


class AIServiceTests(unittest.TestCase):
    def test_perplexity_complete_response_uses_one_request(self):
        client = PerplexityClient(Settings(perplexity_api_key="test"))
        client.session = FakeSession(perplexity_payload("完整報告", "stop"))

        result = client.analyze("system", "prompt")

        self.assertEqual(result.text, "完整報告")
        self.assertTrue(result.completed)
        self.assertFalse(result.continued)
        self.assertEqual(len(client.session.calls), 1)

    def test_perplexity_length_response_is_continued_and_merged(self):
        client = PerplexityClient(Settings(perplexity_api_key="test"))
        client.session = FakeSession(
            perplexity_payload("第一部分尚未完成", "length"),
            perplexity_payload("第二部分與結論", "stop"),
        )

        result = client.analyze("system", "prompt")

        self.assertTrue(result.completed)
        self.assertTrue(result.continued)
        self.assertIn("第一部分尚未完成", result.text)
        self.assertIn("第二部分與結論", result.text)
        self.assertEqual(len(client.session.calls), 2)
        messages = client.session.calls[1][1]["json"]["messages"]
        self.assertEqual(
            messages[-2], {"role": "assistant", "content": "第一部分尚未完成"}
        )

    def test_perplexity_warns_if_continuation_is_also_truncated(self):
        client = PerplexityClient(Settings(perplexity_api_key="test"))
        client.session = FakeSession(
            perplexity_payload("第一段", "length"),
            perplexity_payload("第二段", "max_tokens"),
        )

        result = client.analyze("system", "prompt")

        self.assertFalse(result.completed)
        self.assertIn("可能仍不完整", result.text)

    def test_openai_incomplete_response_is_continued_and_merged(self):
        client = OpenAIClient(Settings(openai_api_key="test"))
        client.session = FakeSession(
            openai_payload("第一部分", "incomplete", "max_output_tokens"),
            openai_payload("後續結論"),
        )

        result = client.analyze("system", "prompt")

        self.assertTrue(result.completed)
        self.assertTrue(result.continued)
        self.assertIn("第一部分", result.text)
        self.assertIn("後續結論", result.text)
        self.assertEqual(len(client.session.calls), 2)
        continuation_input = client.session.calls[1][1]["json"]["input"]
        self.assertEqual(
            continuation_input[1], {"role": "assistant", "content": "第一部分"}
        )

    def test_perplexity_http_errors_are_sanitized(self):
        cases = [(401, "API Key"), (429, "用量限制"), (500, "HTTP 500")]
        for status_code, expected in cases:
            with self.subTest(status_code=status_code):
                client = PerplexityClient(Settings(perplexity_api_key="secret-test-key"))
                client.session = FakeSession(FakeResponse({}, status_code))

                with self.assertRaises(ExternalServiceError) as raised:
                    client.analyze("system", "prompt")

                self.assertIn(expected, raised.exception.public_message)
                self.assertNotIn("secret-test-key", raised.exception.public_message)

    def test_openai_http_errors_are_sanitized(self):
        cases = [(401, "API Key"), (429, "用量限制"), (500, "HTTP 500")]
        for status_code, expected in cases:
            with self.subTest(status_code=status_code):
                client = OpenAIClient(Settings(openai_api_key="secret-test-key"))
                client.session = FakeSession(FakeResponse({}, status_code))

                with self.assertRaises(ExternalServiceError) as raised:
                    client.analyze("system", "prompt")

                self.assertIn(expected, raised.exception.public_message)
                self.assertNotIn("secret-test-key", raised.exception.public_message)

    def test_perplexity_timeout_has_actionable_message(self):
        client = PerplexityClient(Settings(perplexity_api_key="test"))
        client.session = FakeSession(requests.Timeout("simulated timeout"))

        with self.assertRaises(ExternalServiceError) as raised:
            client.analyze("system", "prompt")

        self.assertIn("超過", raised.exception.public_message)
        self.assertIn("sonar", raised.exception.public_message)

    def test_openai_timeout_has_actionable_message(self):
        client = OpenAIClient(Settings(openai_api_key="test"))
        client.session = FakeSession(requests.Timeout("simulated timeout"))

        with self.assertRaises(ExternalServiceError) as raised:
            client.analyze("system", "prompt")

        self.assertIn("逾時", raised.exception.public_message)
