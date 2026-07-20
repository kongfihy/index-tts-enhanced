import Foundation

public struct BackendConfiguration: Equatable, Sendable {
    public let repositoryURL: URL
    public let uvURL: URL
    public let port: Int
    public let modelDirectory: String

    public init(
        repositoryURL: URL,
        uvURL: URL,
        port: Int = 7860,
        modelDirectory: String = "./checkpoints"
    ) {
        self.repositoryURL = repositoryURL
        self.uvURL = uvURL
        self.port = port
        self.modelDirectory = modelDirectory
    }

    public static func systemDefault(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> BackendConfiguration {
        let repositoryPath = environment["INDEXTTS_REPO_PATH"]
            ?? homeDirectory.appendingPathComponent("index-tts").path
        let uvPath = environment["INDEXTTS_UV_PATH"]
            ?? homeDirectory.appendingPathComponent(".local/bin/uv").path
        let port = Int(environment["INDEXTTS_PORT"] ?? "7860") ?? 7860
        let modelDirectory = environment["INDEXTTS_MODEL_DIR"] ?? "./checkpoints"

        return BackendConfiguration(
            repositoryURL: URL(fileURLWithPath: repositoryPath, isDirectory: true),
            uvURL: URL(fileURLWithPath: uvPath),
            port: port,
            modelDirectory: modelDirectory
        )
    }

    public func host(allowLAN: Bool) -> String {
        allowLAN ? "0.0.0.0" : "127.0.0.1"
    }

    public func launchArguments(allowLAN: Bool) -> [String] {
        [
            "run",
            "webui.py",
            "--host", host(allowLAN: allowLAN),
            "--port", String(port),
            "--model_dir", modelDirectory,
        ]
    }

    public var localWebURL: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    public var outputsURL: URL {
        repositoryURL.appendingPathComponent("outputs", isDirectory: true)
    }

    public func cacheRootURL(homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        homeDirectory
            .appendingPathComponent("Library/Caches/IndexTTS", isDirectory: true)
    }

    public func huggingFaceHubCacheURL(homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        cacheRootURL(homeDirectory: homeDirectory)
            .appendingPathComponent("huggingface/hub", isDirectory: true)
    }

    public func runtimeEnvironmentOverrides(
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> [String: String] {
        let cacheRoot = cacheRootURL(homeDirectory: homeDirectory)
        let hubCache = huggingFaceHubCacheURL(homeDirectory: homeDirectory)
        return [
            "INDEXTTS_HF_HUB_CACHE": hubCache.path,
            "HF_HUB_CACHE": hubCache.path,
            "HF_HOME": hubCache.deletingLastPathComponent().path,
            "MPLCONFIGDIR": cacheRoot.appendingPathComponent("matplotlib", isDirectory: true).path,
            "NUMBA_CACHE_DIR": cacheRoot.appendingPathComponent("numba", isDirectory: true).path,
        ]
    }

    public func validationErrors(fileManager: FileManager = .default) -> [String] {
        var errors: [String] = []
        var isDirectory: ObjCBool = false

        if !fileManager.fileExists(atPath: repositoryURL.path, isDirectory: &isDirectory) || !isDirectory.boolValue {
            errors.append("找不到 IndexTTS 目录：\(repositoryURL.path)")
        }
        if !fileManager.isExecutableFile(atPath: uvURL.path) {
            errors.append("找不到可执行的 uv：\(uvURL.path)")
        }
        let webUI = repositoryURL.appendingPathComponent("webui.py").path
        if !fileManager.fileExists(atPath: webUI) {
            errors.append("找不到 webui.py：\(webUI)")
        }
        let modelPath = repositoryURL.appendingPathComponent(modelDirectory).standardized.path
        if !fileManager.fileExists(atPath: modelPath, isDirectory: &isDirectory) || !isDirectory.boolValue {
            errors.append("找不到模型目录：\(modelPath)")
        }
        if !(1...65535).contains(port) {
            errors.append("端口无效：\(port)")
        }
        return errors
    }
}
