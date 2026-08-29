import json
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from paperBotV2.arxiv_daily import arxiv
from paperBotV2.arxiv_daily import arxiv_feishu_msg
from paperBotV2.arxiv_daily import generate_arxiv_html


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 8, 29, tzinfo=tz)


class ArxivPipelineTests(unittest.TestCase):
    def test_process_fails_before_fetch_when_api_key_is_missing(self):
        status = Mock()
        status.data = {"stage": "validate"}
        with (
            patch.object(arxiv, "DEEPSEEK_API_KEY", ""),
            patch.object(arxiv, "ArxivDailyStatus", return_value=status),
            patch.object(arxiv, "get_papers_from_all_categories") as fetch,
        ):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                arxiv.process_papers()

        fetch.assert_not_called()
        status.mark_failed.assert_called_once()

    def test_zero_success_rough_ranking_fails_when_papers_exist(self):
        papers = {"1234.5678": {"arxiv_id": "1234.5678", "title": "Paper"}}
        with patch.object(arxiv, "rough_rank_papers", return_value=([], [])):
            with self.assertRaisesRegex(RuntimeError, "Rough ranking failed"):
                arxiv.perform_rough_ranking(papers)

    def test_zero_success_fine_ranking_fails_when_candidates_exist(self):
        paper = {"arxiv_id": "1234.5678", "title": "Paper"}
        with patch.object(arxiv, "fine_rank_papers", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "Fine ranking failed"):
                arxiv.perform_fine_ranking([paper], {paper["arxiv_id"]: paper})

    def test_empty_daily_result_is_written_as_valid_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_module_path = os.path.join(temp_dir, "arxiv.py")
            with (
                patch.object(arxiv, "__file__", fake_module_path),
                patch.object(arxiv, "datetime", FixedDateTime),
            ):
                self.assertTrue(arxiv.save_results_to_json({}))

            daily_path = Path(temp_dir, "data", "20260829.json")
            self.assertEqual(json.loads(daily_path.read_text(encoding="utf-8")), {})

    def test_malformed_aggregate_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir, "data")
            data_dir.mkdir()
            aggregate_path = data_dir / "results.json"
            original = b'{"existing": '
            aggregate_path.write_bytes(original)

            with (
                patch.object(arxiv, "__file__", os.path.join(temp_dir, "arxiv.py")),
                patch.object(arxiv, "datetime", FixedDateTime),
            ):
                with self.assertRaisesRegex(ValueError, "results.json"):
                    arxiv.save_results_to_json({"new": {"title": "Paper"}})

            self.assertEqual(aggregate_path.read_bytes(), original)
            self.assertFalse((data_dir / "20260829.json").exists())

    def test_non_object_aggregate_is_rejected_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir, "data")
            data_dir.mkdir()
            aggregate_path = data_dir / "results.json"
            original = b'[{"title": "not an object"}]'
            aggregate_path.write_bytes(original)

            with (
                patch.object(arxiv, "__file__", os.path.join(temp_dir, "arxiv.py")),
                patch.object(arxiv, "datetime", FixedDateTime),
            ):
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    arxiv.save_results_to_json({"new": {"title": "Paper"}})

            self.assertEqual(aggregate_path.read_bytes(), original)
            self.assertFalse((data_dir / "20260829.json").exists())

    def test_atomic_aggregate_write_preserves_previous_file_on_dump_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir, "data")
            data_dir.mkdir()
            aggregate_path = data_dir / "results.json"
            existing = {"existing": {"title": "Existing"}}
            original = json.dumps(existing).encode("utf-8")
            aggregate_path.write_bytes(original)
            real_dump = json.dump

            def fail_for_aggregate(value, file_obj, *args, **kwargs):
                if set(value) == {"existing", "new"}:
                    raise OSError("simulated interrupted aggregate write")
                return real_dump(value, file_obj, *args, **kwargs)

            with (
                patch.object(arxiv, "__file__", os.path.join(temp_dir, "arxiv.py")),
                patch.object(arxiv, "datetime", FixedDateTime),
                patch.object(arxiv.json, "dump", side_effect=fail_for_aggregate),
            ):
                with self.assertRaisesRegex(OSError, "interrupted aggregate write"):
                    arxiv.save_results_to_json({"new": {"title": "Paper"}})

            self.assertEqual(aggregate_path.read_bytes(), original)


class HtmlCliTests(unittest.TestCase):
    def test_empty_json_is_valid_and_generates_html(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir, "data")
            data_dir.mkdir()
            (data_dir / "20260829.json").write_text("{}", encoding="utf-8")

            with (
                patch.object(generate_arxiv_html, "__file__", os.path.join(temp_dir, "script.py")),
                patch.object(sys, "argv", ["generate_arxiv_html.py", "--date", "20260829"]),
                patch.object(generate_arxiv_html, "generate_html", return_value="output.html") as generate,
            ):
                self.assertEqual(generate_arxiv_html.main(), 0)

            self.assertEqual(generate.call_args.args[0], [])

    def test_malformed_json_returns_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir, "data")
            data_dir.mkdir()
            (data_dir / "20260829.json").write_text("[]", encoding="utf-8")

            with (
                patch.object(generate_arxiv_html, "__file__", os.path.join(temp_dir, "script.py")),
                patch.object(sys, "argv", ["generate_arxiv_html.py", "--date", "20260829"]),
                patch.object(generate_arxiv_html, "generate_html") as generate,
            ):
                self.assertEqual(generate_arxiv_html.main(), 1)

            generate.assert_not_called()

    def test_conflicting_arguments_return_usage_error(self):
        with patch.object(sys, "argv", ["generate_arxiv_html.py", "--all", "--date", "20260829"]):
            self.assertEqual(generate_arxiv_html.main(), 2)


class FeishuCardTests(unittest.TestCase):
    def setUp(self):
        self.paper = {
            "title": "A portable card",
            "translation": "跨租户卡片",
            "rerank_relevance_score": 8,
            "summary": "Summary",
            "url": "https://arxiv.org/abs/1234.5678",
        }

    def test_card_is_raw_and_multi_webhook_delivery_is_preserved(self):
        response = Mock(ok=True, status_code=200, text='{"StatusCode": 0}')
        response.json.return_value = {"StatusCode": 0}
        with patch.object(arxiv_feishu_msg.requests, "post", return_value=response) as post:
            arxiv_feishu_msg.send_papers_to_feishu(
                [self.paper],
                ["https://example.test/one", "https://example.test/two"],
            )

        self.assertEqual(post.call_count, 2)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["msg_type"], "interactive")
        self.assertIsInstance(payload["card"], dict)
        self.assertNotIn("template_id", json.dumps(payload))

    def test_card_rejects_non_http_paper_url(self):
        paper = dict(self.paper, url="javascript:alert(1)")
        card = arxiv_feishu_msg.build_feishu_card([paper], "2026-08-29")
        content = card["elements"][0]["text"]["content"]
        self.assertIn("(https://arxiv.org)", content)
        self.assertNotIn("javascript:", content)

    def test_business_error_from_any_webhook_fails_delivery(self):
        response = Mock(ok=True, status_code=200, text='{"code": 19001}')
        response.json.return_value = {"code": 19001, "msg": "invalid token"}
        with patch.object(arxiv_feishu_msg.requests, "post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "飞书推送存在失败"):
                arxiv_feishu_msg.send_papers_to_feishu(
                    [self.paper], ["https://example.test/hook"]
                )

    def test_http_failure_does_not_log_response_body_or_webhook_token(self):
        secret = "secret-webhook-token"
        webhook = f"https://open.feishu.cn/open-apis/bot/v2/hook/{secret}"
        response = Mock(ok=False, status_code=403, text=f"rejected {secret}")
        output = io.StringIO()

        with (
            patch.object(arxiv_feishu_msg.requests, "post", return_value=response),
            redirect_stdout(output),
        ):
            with self.assertRaises(RuntimeError) as caught:
                arxiv_feishu_msg.send_papers_to_feishu([self.paper], [webhook])

        combined = output.getvalue() + str(caught.exception)
        self.assertNotIn(secret, combined)
        self.assertNotIn(webhook, combined)
        self.assertNotIn("rejected", combined)

    def test_business_failure_does_not_log_response_message_or_webhook_token(self):
        secret = "secret-webhook-token"
        webhook = f"https://open.feishu.cn/open-apis/bot/v2/hook/{secret}"
        response = Mock(ok=True, status_code=200, text=f'{{"code": 19001, "msg": "{secret}"}}')
        response.json.return_value = {"code": 19001, "msg": f"invalid {secret}"}
        output = io.StringIO()

        with (
            patch.object(arxiv_feishu_msg.requests, "post", return_value=response),
            redirect_stdout(output),
        ):
            with self.assertRaises(RuntimeError) as caught:
                arxiv_feishu_msg.send_papers_to_feishu([self.paper], [webhook])

        combined = output.getvalue() + str(caught.exception)
        self.assertNotIn(secret, combined)
        self.assertNotIn(webhook, combined)
        self.assertNotIn("invalid", combined)

    def test_request_exception_does_not_log_exception_with_webhook_token(self):
        secret = "secret-webhook-token"
        webhook = f"https://open.feishu.cn/open-apis/bot/v2/hook/{secret}"
        error = arxiv_feishu_msg.requests.RequestException(f"failed request to {webhook}")
        output = io.StringIO()

        with (
            patch.object(arxiv_feishu_msg.requests, "post", side_effect=error),
            redirect_stdout(output),
        ):
            with self.assertRaises(RuntimeError) as caught:
                arxiv_feishu_msg.send_papers_to_feishu([self.paper], [webhook])

        combined = output.getvalue() + str(caught.exception)
        self.assertNotIn(secret, combined)
        self.assertNotIn(webhook, combined)
        self.assertNotIn("failed request", combined)

    def test_main_fails_when_papers_need_delivery_but_webhook_is_missing(self):
        today_file = f"{datetime.now().strftime('%Y%m%d')}.json"
        deliverable = dict(self.paper, is_fine_ranked=True)
        with (
            patch.object(arxiv_feishu_msg, "FEISHU_URLS", []),
            patch.object(arxiv_feishu_msg, "get_latest_json_file", return_value=today_file),
            patch.object(arxiv_feishu_msg, "load_paper_data", return_value=[deliverable]),
        ):
            self.assertEqual(arxiv_feishu_msg.main(), 1)

    def test_main_allows_no_selected_papers_without_webhook(self):
        today_file = f"{datetime.now().strftime('%Y%m%d')}.json"
        with (
            patch.object(arxiv_feishu_msg, "FEISHU_URLS", []),
            patch.object(arxiv_feishu_msg, "get_latest_json_file", return_value=today_file),
            patch.object(arxiv_feishu_msg, "load_paper_data", return_value=[self.paper]),
        ):
            self.assertEqual(arxiv_feishu_msg.main(), 0)


class WorkflowConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.full_workflow = (repo_root / ".github/workflows/arxiv_daily_full.yml").read_text(
            encoding="utf-8"
        )
        cls.recovery_workflow = (
            repo_root / ".github/workflows/arxiv_daily_recovery.yml"
        ).read_text(encoding="utf-8")

    def test_full_and_recovery_share_non_cancelling_concurrency_group(self):
        for workflow in (self.full_workflow, self.recovery_workflow):
            self.assertIn(
                "group: paperbot-main-writer-${{ github.repository }}",
                workflow,
            )
            self.assertIn("cancel-in-progress: false", workflow)

    def test_pages_success_is_marked_only_after_deployment(self):
        for workflow in (self.full_workflow, self.recovery_workflow):
            deploy_position = workflow.index("- name: Deploy to GitHub Pages")
            mark_position = workflow.index("- name: Mark Pages deployment successful")
            status_commit_position = workflow.index("- name: Commit successful deployment status")
            self.assertLess(deploy_position, mark_position)
            self.assertLess(mark_position, status_commit_position)
            self.assertIn("update_today_output(html_generated=False)", workflow)


if __name__ == "__main__":
    unittest.main()
