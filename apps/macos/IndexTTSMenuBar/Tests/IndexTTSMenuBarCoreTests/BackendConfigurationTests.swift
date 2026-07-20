import Foundation
import Testing
@testable import IndexTTSMenuBarCore

struct BackendConfigurationTests {
    @Test func localAndLANHosts() {
        let config = makeConfiguration()
        #expect(config.host(allowLAN: false) == "127.0.0.1")
        #expect(config.host(allowLAN: true) == "0.0.0.0")
    }

    @Test func launchArgumentsAreStable() {
        let config = makeConfiguration()
        #expect(config.launchArguments(allowLAN: false) == [
            "run", "webui.py",
            "--host", "127.0.0.1",
            "--port", "7860",
            "--model_dir", "./checkpoints",
        ])
    }

    @Test func runtimeCacheEnvironmentUsesStableMacOSLocation() {
        let config = makeConfiguration()
        let home = URL(fileURLWithPath: "/Users/example", isDirectory: true)
        let environment = config.runtimeEnvironmentOverrides(homeDirectory: home)

        #expect(environment["INDEXTTS_HF_HUB_CACHE"] == "/Users/example/Library/Caches/IndexTTS/huggingface/hub")
        #expect(environment["HF_HUB_CACHE"] == "/Users/example/Library/Caches/IndexTTS/huggingface/hub")
        #expect(environment["HF_HOME"] == "/Users/example/Library/Caches/IndexTTS/huggingface")
        #expect(environment["MPLCONFIGDIR"] == "/Users/example/Library/Caches/IndexTTS/matplotlib")
        #expect(environment["NUMBA_CACHE_DIR"] == "/Users/example/Library/Caches/IndexTTS/numba")
    }

    @Test func environmentOverridesDefaults() {
        let config = BackendConfiguration.systemDefault(
            environment: [
                "INDEXTTS_REPO_PATH": "/tmp/custom-index-tts",
                "INDEXTTS_UV_PATH": "/tmp/custom-uv",
                "INDEXTTS_PORT": "9000",
                "INDEXTTS_MODEL_DIR": "models",
            ],
            homeDirectory: URL(fileURLWithPath: "/Users/example", isDirectory: true)
        )
        #expect(config.repositoryURL.path == "/tmp/custom-index-tts")
        #expect(config.uvURL.path == "/tmp/custom-uv")
        #expect(config.port == 9000)
        #expect(config.modelDirectory == "models")
        #expect(config.localWebURL.absoluteString == "http://127.0.0.1:9000")
    }

    private func makeConfiguration() -> BackendConfiguration {
        BackendConfiguration(
            repositoryURL: URL(fileURLWithPath: "/Users/example/index-tts", isDirectory: true),
            uvURL: URL(fileURLWithPath: "/Users/example/.local/bin/uv"),
            port: 7860,
            modelDirectory: "./checkpoints"
        )
    }
}
