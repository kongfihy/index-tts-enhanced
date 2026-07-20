import tempfile
import unittest
from pathlib import Path

from indextts.utils.hf_cache import (
    configure_huggingface_environment,
    hub_cache_candidates,
    load_pretrained_offline_first,
    persistent_hub_cache,
)


class HuggingFaceCacheTests(unittest.TestCase):
    def test_macos_default_is_persistent_user_cache(self):
        home = Path("/Users/example")
        self.assertEqual(
            persistent_hub_cache(home=home, platform="darwin"),
            home / "Library/Caches/IndexTTS/huggingface/hub",
        )

    def test_candidates_keep_legacy_temp_cache_as_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidates = hub_cache_candidates(
                project_root=root / "repo",
                environ={},
                home=root / "home",
                platform="darwin",
                temp_root=root / "private-tmp",
            )
            self.assertEqual(
                candidates[0],
                (root / "home/Library/Caches/IndexTTS/huggingface/hub").resolve(),
            )
            self.assertIn(
                (root / "private-tmp/indextts-cache/huggingface/hub").resolve(),
                candidates,
            )

    def test_configure_sets_cache_before_third_party_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = {}
            cache = configure_huggingface_environment(
                project_root=root / "repo",
                environ=environment,
                home=root / "home",
                platform="darwin",
            )
            self.assertTrue(cache.is_dir())
            self.assertEqual(environment["INDEXTTS_HF_HUB_CACHE"], str(cache))
            self.assertEqual(environment["HF_HUB_CACHE"], str(cache))
            self.assertEqual(environment["HF_HOME"], str(cache.parent))

    def test_configure_propagates_offline_mode_to_huggingface_libraries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = {"INDEXTTS_OFFLINE": "1"}

            configure_huggingface_environment(
                project_root=root / "repo",
                environ=environment,
                home=root / "home",
                platform="darwin",
            )

            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")

    def test_loader_uses_second_local_cache_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary = root / "primary"
            fallback = root / "repo/checkpoints/hf_cache"
            primary.mkdir(parents=True)
            fallback.mkdir(parents=True)
            (fallback / "cached-model").mkdir()
            fallback = fallback.resolve()
            calls = []

            def fake_loader(model_id, cache_dir, local_files_only=False):
                calls.append((Path(cache_dir), local_files_only))
                if Path(cache_dir) == fallback and local_files_only:
                    return "local-model"
                if not local_files_only:
                    self.fail("network fallback should not be used")
                raise OSError("not cached here")

            result = load_pretrained_offline_first(
                fake_loader,
                "example/model",
                project_root=root / "repo",
                environ={"INDEXTTS_HF_HUB_CACHE": str(primary)},
            )

            self.assertEqual(result, "local-model")
            self.assertEqual(calls[-1], (fallback, True))

    def test_offline_mode_fails_without_network_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary = root / "primary"
            primary.mkdir(parents=True)
            (primary / "something").touch()
            calls = []

            def fake_loader(model_id, cache_dir, local_files_only=False):
                calls.append(local_files_only)
                raise OSError("missing")

            with self.assertRaisesRegex(RuntimeError, "offline mode"):
                load_pretrained_offline_first(
                    fake_loader,
                    "example/model",
                    project_root=root / "repo",
                    environ={
                        "INDEXTTS_HF_HUB_CACHE": str(primary),
                        "INDEXTTS_OFFLINE": "1",
                    },
                )
            self.assertTrue(calls)
            self.assertTrue(all(calls))


if __name__ == "__main__":
    unittest.main()
