import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conf_daily


class CandidateSelectionTests(unittest.TestCase):
    def test_selects_only_unsent_papers_with_abstracts(self):
        papers = [
            {
                "paper_name": "Search ranking without an abstract",
                "paper_url": "https://example.com/empty",
                "paper_abstract": "",
            },
            {
                "paper_name": "Recommendation candidate generation",
                "paper_url": "https://example.com/sent",
                "paper_abstract": "Already sent abstract.",
            },
            {
                "paper_name": "Recommendation ranking",
                "paper_url": "https://example.com/new",
                "paper_abstract": "New abstract.",
            },
        ]
        sent_id = conf_daily.stable_paper_id("kdd2025", papers[1])

        selected = conf_daily.select_candidates(
            {"kdd2025": papers},
            sent_paper_ids={sent_id},
            limits=10,
            confs=["kdd"],
            start_year=2025,
            current_year=2025,
        )

        self.assertEqual([item["paper"] for item in selected], [papers[2]])
        self.assertEqual(
            selected[0]["paper_id"],
            conf_daily.stable_paper_id("kdd2025", papers[2]),
        )

    def test_saved_state_excludes_paper_on_next_selection(self):
        paper = {
            "paper_name": "Industrial recommendation",
            "paper_url": "https://example.com/paper",
            "paper_abstract": "Abstract.",
        }
        paper_id = conf_daily.stable_paper_id("www2025", paper)

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "push_state.json"
            conf_daily.save_state({paper_id}, state_path)
            state = conf_daily.load_state(state_path)

        selected = conf_daily.select_candidates(
            {"www2025": [paper]},
            sent_paper_ids=state["sent_paper_ids"],
            limits=10,
            confs=["www"],
            start_year=2025,
            current_year=2025,
        )
        self.assertEqual(selected, [])


class StateSafetyTests(unittest.TestCase):
    def test_later_send_failure_preserves_earlier_paper_checkpoint(self):
        results = {
            "kdd2025": [
                {
                    "paper_name": "Recommendation ranking",
                    "paper_url": "https://example.com/one",
                    "paper_abstract": "First abstract.",
                    "paper_authors": ["A"],
                },
                {
                    "paper_name": "Search ranking",
                    "paper_url": "https://example.com/two",
                    "paper_abstract": "Second abstract.",
                    "paper_authors": ["B"],
                },
            ]
        }
        initial_state = {"version": 1, "sent_paper_ids": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_path = temp_path / "results.json"
            state_path = temp_path / "push_state.json"
            results_path.write_text(json.dumps(results), encoding="utf-8")
            state_path.write_text(json.dumps(initial_state), encoding="utf-8")
            args = argparse.Namespace(
                results=results_path,
                state=state_path,
                limits=2,
                push_interval=0,
                start_year=2012,
                confs="kdd",
                deepseek_model="deepseek-chat",
                dry_run=False,
            )

            with mock.patch.dict(
                conf_daily.os.environ,
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "FEISHU_URL": "https://example.com/webhook",
                },
                clear=False,
            ), mock.patch.object(
                conf_daily,
                "translate_with_deepseek",
                return_value=["译文一", "译文二"],
            ), mock.patch.object(
                conf_daily,
                "send_feishu_message",
                side_effect=[None, conf_daily.DeliveryError("send failed")],
            ):
                with self.assertRaises(conf_daily.DeliveryError):
                    conf_daily.run(args)

            saved_state = conf_daily.load_state(state_path)
            first_paper_id = conf_daily.stable_paper_id(
                "kdd2025", results["kdd2025"][0]
            )
            second_paper_id = conf_daily.stable_paper_id(
                "kdd2025", results["kdd2025"][1]
            )
            self.assertIn(first_paper_id, saved_state["sent_paper_ids"])
            self.assertNotIn(second_paper_id, saved_state["sent_paper_ids"])

    def test_paper_is_not_checkpointed_until_every_webhook_succeeds(self):
        paper = {
            "paper_name": "Industrial recommendation",
            "paper_url": "https://example.com/paper",
            "paper_abstract": "Abstract.",
            "paper_authors": ["A"],
        }
        results = {"kdd2025": [paper]}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_path = temp_path / "results.json"
            state_path = temp_path / "push_state.json"
            results_path.write_text(json.dumps(results), encoding="utf-8")
            state_path.write_text(
                json.dumps({"version": 1, "sent_paper_ids": []}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                results=results_path,
                state=state_path,
                limits=1,
                push_interval=0,
                start_year=2012,
                confs="kdd",
                deepseek_model="deepseek-chat",
                dry_run=False,
            )
            successful_response = mock.Mock()
            successful_response.json.return_value = {"code": 0}

            with mock.patch.dict(
                conf_daily.os.environ,
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "FEISHU_URL": (
                        "https://open.feishu.cn/hook/first,"
                        "https://open.feishu.cn/hook/second"
                    ),
                },
                clear=False,
            ), mock.patch.object(
                conf_daily,
                "translate_with_deepseek",
                return_value=["译文"],
            ), mock.patch.object(
                conf_daily.requests,
                "post",
                side_effect=[
                    successful_response,
                    conf_daily.requests.ConnectionError("second endpoint failed"),
                ],
            ):
                with self.assertRaises(conf_daily.DeliveryError):
                    conf_daily.run(args)

            saved_state = conf_daily.load_state(state_path)
            self.assertEqual(saved_state["sent_paper_ids"], [])


class FeishuDeliveryTests(unittest.TestCase):
    @mock.patch.object(conf_daily.requests, "post")
    def test_sends_card_as_raw_json_object(self, post):
        response = mock.Mock()
        response.json.return_value = {"code": 0, "msg": "success"}
        post.return_value = response

        conf_daily.send_feishu_message(
            "title", "content", ["https://example.com/webhook"]
        )

        response.raise_for_status.assert_called_once_with()
        payload = post.call_args.kwargs["json"]
        self.assertIsInstance(payload["card"], dict)
        self.assertEqual(payload["msg_type"], "interactive")

    @mock.patch.object(conf_daily.requests, "post")
    def test_rejects_nonzero_feishu_business_code(self, post):
        response = mock.Mock()
        response.json.return_value = {
            "code": 19001,
            "msg": "bad webhook https://example.com/hook/SECRET_TOKEN",
        }
        post.return_value = response

        with self.assertRaises(conf_daily.DeliveryError) as raised:
            conf_daily.send_feishu_message(
                "title", "content", ["https://example.com/webhook"]
            )

        self.assertNotIn("SECRET_TOKEN", str(raised.exception))
        self.assertNotIn("bad webhook", str(raised.exception))
        self.assertIn("code=19001", str(raised.exception))

    @mock.patch.object(conf_daily.requests, "post")
    def test_request_error_does_not_leak_webhook_or_exception_url(self, post):
        secret_url = "https://open.feishu.cn/open-apis/bot/v2/hook/SECRET_TOKEN"
        post.side_effect = requests_error = conf_daily.requests.ConnectionError(
            f"connection failed for {secret_url}"
        )

        with self.assertRaises(conf_daily.DeliveryError) as raised:
            conf_daily.send_feishu_message("title", "content", [secret_url])

        message = str(raised.exception)
        self.assertNotIn("SECRET_TOKEN", message)
        self.assertNotIn(str(requests_error), message)
        self.assertIn("端点 1/1", message)
        self.assertIn("open.feishu.cn", message)
        self.assertIn("ConnectionError", message)

    @mock.patch.object(conf_daily.requests, "post")
    def test_http_error_only_reports_host_and_status(self, post):
        secret_url = "https://open.feishu.cn/open-apis/bot/v2/hook/SECRET_TOKEN"
        response = mock.Mock(status_code=403)
        response.raise_for_status.side_effect = conf_daily.requests.HTTPError(
            f"403 Client Error for url: {secret_url}", response=response
        )
        post.return_value = response

        with self.assertRaises(conf_daily.DeliveryError) as raised:
            conf_daily.send_feishu_message("title", "content", [secret_url])

        message = str(raised.exception)
        self.assertNotIn("SECRET_TOKEN", message)
        self.assertNotIn(secret_url, message)
        self.assertIn("open.feishu.cn", message)
        self.assertIn("status=403", message)


if __name__ == "__main__":
    unittest.main()
