from __future__ import annotations

import json
import unittest

from edusense.local_ai import LocalModelError, LocalTeachingAssistant


class LocalAssistantTests(unittest.TestCase):
    def test_local_model_returns_three_validated_suggestions(self) -> None:
        body = {
            "response": json.dumps({
                "suggestions": [
                    {"title": "Check in", "action": "Use a one-question poll.", "reason": "Gets a quick learner response."},
                    {"title": "Pair discussion", "action": "Invite a short paired explanation.", "reason": "Offers another participation route."},
                    {"title": "Pause", "action": "Restate the next task step.", "reason": "Makes the transition clear."},
                ]
            })
        }

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return body

        calls: list[dict[str, object]] = []

        def fake_post(url: str, **kwargs: object) -> Response:
            calls.append({"url": url, **kwargs})
            return Response()

        assistant = LocalTeachingAssistant("my-local-model", http_post=fake_post)
        suggestions = assistant.suggest({"mean_activity_signal": 48.0})
        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions[0].title, "Check in")
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:11434/api/generate")
        self.assertEqual(calls[0]["json"]["stream"], False)

    def test_remote_endpoint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LocalTeachingAssistant("model", endpoint="https://example.com/api/generate")

    def test_invalid_response_is_rejected(self) -> None:
        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"response": "{\"suggestions\": []}"}

        assistant = LocalTeachingAssistant("model", http_post=lambda *args, **kwargs: Response())
        with self.assertRaises(LocalModelError):
            assistant.suggest({"mean_activity_signal": 20.0})


if __name__ == "__main__":
    unittest.main()
