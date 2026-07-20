import AppKit
import Foundation
import IndexTTSMenuBarCore
import ServiceManagement

@MainActor
final class BackendController: ObservableObject {
    enum Status: Equatable {
        case stopped
        case starting
        case running
        case external
        case stopping
        case failed(String)

        var title: String {
            switch self {
            case .stopped: return "未运行"
            case .starting: return "模型加载中"
            case .running: return "已运行"
            case .external: return "已运行（外部终端）"
            case .stopping: return "正在停止"
            case .failed: return "启动失败"
            }
        }

    }

    enum Activity: Equatable {
        case idle
        case queued(Int)
        case generating(Int, String, Double)
        case taskFailed(Int)

        var title: String {
            switch self {
            case .idle: return "当前空闲"
            case .queued(let count): return "排队中 \(count)"
            case .generating(let count, let project, let progress):
                return "正在生成 \(count) · \(project) · \(Int(progress * 100))%"
            case .taskFailed(let count): return "最近有 \(count) 个失败任务"
            }
        }
    }

    private struct TaskCenterStatus: Decodable {
        let active_jobs: Int
        let queued_jobs: Int
        let failed_jobs: Int
        let current_job: TaskCenterJob?
    }

    private struct TaskCenterJob: Decodable {
        let job_id: String
        let project_name: String
        let progress: Double
        let stage: String
    }

    @Published private(set) var status: Status = .stopped
    @Published private(set) var activity: Activity = .idle
    @Published private(set) var currentJobID: String?
    @Published var allowLAN: Bool {
        didSet {
            defaults.set(allowLAN, forKey: Keys.allowLAN)
            if status == .running || status == .starting {
                message = "网络设置已改变，重新启动服务后生效"
            }
        }
    }
    @Published var startBackendWithApp: Bool {
        didSet { defaults.set(startBackendWithApp, forKey: Keys.startBackendWithApp) }
    }
    @Published private(set) var launchAtLogin: Bool = false
    @Published private(set) var message: String = ""

    let configuration: BackendConfiguration

    private enum Keys {
        static let allowLAN = "allowLAN"
        static let startBackendWithApp = "startBackendWithApp"
    }

    private let defaults: UserDefaults
    private let fileManager: FileManager
    private var process: Process?
    private var logHandle: FileHandle?
    private var healthTask: Task<Void, Never>?
    private var requestedStop = false
    private var launchStartedAt: Date?
    private var backendLogStartOffset: UInt64 = 0

    private var appSupportURL: URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/IndexTTSMenuBar", isDirectory: true)
    }

    private var logsURL: URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/IndexTTSMenuBar", isDirectory: true)
    }

    private var backendLogURL: URL { logsURL.appendingPathComponent("backend.log") }
    private var launcherLogURL: URL { logsURL.appendingPathComponent("launcher.log") }
    private var pidURL: URL { appSupportURL.appendingPathComponent("backend.pid") }

    init(
        configuration: BackendConfiguration = .systemDefault(),
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default
    ) {
        self.configuration = configuration
        self.defaults = defaults
        self.fileManager = fileManager
        self.allowLAN = defaults.bool(forKey: Keys.allowLAN)
        self.startBackendWithApp = defaults.bool(forKey: Keys.startBackendWithApp)
        self.launchAtLogin = SMAppService.mainApp.status == .enabled

        prepareDirectories()
        startHealthMonitoring()

        if startBackendWithApp {
            Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(600))
                guard let self else { return }
                await self.refreshHealth()
                if self.status == .stopped || self.isFailureStatus {
                    self.start()
                }
            }
        }
    }

    deinit {
        healthTask?.cancel()
    }

    func start() {
        guard status != .starting, status != .running, status != .external else { return }
        let errors = configuration.validationErrors(fileManager: fileManager)
        guard errors.isEmpty else {
            status = .failed(errors.joined(separator: "\n"))
            message = errors.joined(separator: "；")
            appendLauncherLog("启动前检查失败：\(errors.joined(separator: " | "))")
            return
        }

        prepareDirectories()
        requestedStop = false
        launchStartedAt = Date()
        status = .starting
        message = "模型加载中，通常需要 1～2 分钟，请勿重复启动"
        appendLauncherLog("启动后端，host=\(configuration.host(allowLAN: allowLAN)), port=\(configuration.port)")

        do {
            if !fileManager.fileExists(atPath: backendLogURL.path) {
                fileManager.createFile(atPath: backendLogURL.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: backendLogURL)
            backendLogStartOffset = try handle.seekToEnd()
            logHandle = handle

            let backend = Process()
            backend.executableURL = configuration.uvURL
            backend.arguments = configuration.launchArguments(allowLAN: allowLAN)
            backend.currentDirectoryURL = configuration.repositoryURL
            backend.standardOutput = handle
            backend.standardError = handle

            var environment = ProcessInfo.processInfo.environment
            let requiredPaths = [
                configuration.uvURL.deletingLastPathComponent().path,
                "/opt/homebrew/bin",
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
            ]
            let currentPath = environment["PATH"] ?? ""
            environment["PATH"] = (requiredPaths + [currentPath])
                .filter { !$0.isEmpty }
                .joined(separator: ":")
            environment["PYTHONUNBUFFERED"] = "1"
            for (name, value) in configuration.runtimeEnvironmentOverrides(
                homeDirectory: fileManager.homeDirectoryForCurrentUser
            ) {
                environment[name] = value
            }
            backend.environment = environment

            backend.terminationHandler = { [weak self] terminated in
                let code = terminated.terminationStatus
                Task { @MainActor [weak self] in
                    self?.handleTermination(exitCode: code)
                }
            }

            try backend.run()
            process = backend
            try String(backend.processIdentifier).write(to: pidURL, atomically: true, encoding: .utf8)
            appendLauncherLog("后端进程已创建，pid=\(backend.processIdentifier)")
        } catch {
            status = .failed(error.localizedDescription)
            message = "无法启动：\(error.localizedDescription)"
            appendLauncherLog("启动失败：\(error)")
            closeLogHandle()
        }
    }

    func stop() {
        guard status != .external else {
            message = "这是由终端启动的服务，请先在原终端停止"
            return
        }
        guard status != .stopping, status != .stopped else { return }
        requestedStop = true
        status = .stopping
        message = "正在停止服务并释放模型内存"
        appendLauncherLog("请求停止后端")

        if let process, process.isRunning {
            process.terminate()
            scheduleForceStop(pid: process.processIdentifier)
            return
        }

        guard let pid = storedPID(), processExists(pid) else {
            cleanupAfterStop()
            return
        }

        Darwin.kill(pid, SIGTERM)
        scheduleForceStop(pid: pid)
    }

    func restart() {
        guard status != .external else {
            message = "外部服务不能由菜单栏重启"
            return
        }
        Task { [weak self] in
            guard let self else { return }
            self.stop()
            for _ in 0..<20 {
                try? await Task.sleep(for: .milliseconds(250))
                if self.status == .stopped { break }
            }
            self.start()
        }
    }

    func openWebUI() {
        NSWorkspace.shared.open(configuration.localWebURL)
    }

    func openTaskCenter() {
        NSWorkspace.shared.open(URL(string: "http://127.0.0.1:7861")!)
    }

    func cancelCurrentTask() {
        guard let currentJobID else { return }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:7861/api/jobs/\(currentJobID)/cancel")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 2
        Task {
            do {
                _ = try await URLSession.shared.data(for: request)
                message = "已请求安全取消当前任务"
            } catch {
                message = "取消请求失败：\(error.localizedDescription)"
            }
        }
    }

    func openOutputs() {
        try? fileManager.createDirectory(at: configuration.outputsURL, withIntermediateDirectories: true)
        NSWorkspace.shared.open(configuration.outputsURL)
    }

    func openLogs() {
        prepareDirectories()
        NSWorkspace.shared.open(logsURL)
    }

    func copyDiagnosticInfo() {
        let text = """
        IndexTTS MenuBar
        状态：\(status.title)
        仓库：\(configuration.repositoryURL.path)
        uv：\(configuration.uvURL.path)
        地址：\(configuration.localWebURL.absoluteString)
        局域网：\(allowLAN ? "开启" : "关闭")
        日志：\(backendLogURL.path)
        信息：\(message)
        """
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        message = "诊断信息已复制"
    }

    func setLaunchAtLogin(_ enabled: Bool) {
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            launchAtLogin = SMAppService.mainApp.status == .enabled
            message = launchAtLogin ? "已设置登录后启动菜单栏" : "已关闭登录后启动"
        } catch {
            launchAtLogin = SMAppService.mainApp.status == .enabled
            message = "无法修改登录项：\(error.localizedDescription)"
            appendLauncherLog("登录项设置失败：\(error)")
        }
    }

    func quit(stopBackend: Bool) {
        if stopBackend, status != .stopped {
            stop()
            Task {
                for _ in 0..<24 {
                    try? await Task.sleep(for: .milliseconds(250))
                    if status == .stopped { break }
                }
                NSApplication.shared.terminate(nil)
            }
        } else {
            NSApplication.shared.terminate(nil)
        }
    }

    private var isFailureStatus: Bool {
        if case .failed = status { return true }
        return false
    }

    private func startHealthMonitoring() {
        healthTask?.cancel()
        healthTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refreshHealth()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private func refreshHealth() async {
        var request = URLRequest(url: configuration.localWebURL)
        request.timeoutInterval = 1.2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse, (200...499).contains(http.statusCode) {
                if status != .stopping {
                    if ownedProcessIsRunning() || storedProcessIsRunning() {
                        status = .running
                        launchStartedAt = nil
                        message = allowLAN ? "服务可用，局域网访问已开启" : "服务可用，仅本机访问"
                    } else {
                        status = .external
                        message = "检测到终端启动的服务；停止原服务后可由菜单栏接管"
                    }
                    await refreshTaskActivity()
                }
                return
            }
        } catch {
            // A failed probe is expected while the model is loading or stopped.
        }

        if status == .starting, ownedProcessIsRunning() || storedProcessIsRunning() {
            updateLoadingMessage()
        }

        if (status == .running || status == .external), !ownedProcessIsRunning(), !storedProcessIsRunning() {
            status = .stopped
            launchStartedAt = nil
            message = "服务已停止"
        }
    }

    private func refreshTaskActivity() async {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:7861/api/status")!)
        request.timeoutInterval = 1
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }
            let summary = try JSONDecoder().decode(TaskCenterStatus.self, from: data)
            currentJobID = summary.current_job?.job_id
            if let job = summary.current_job, summary.active_jobs > 0 {
                activity = .generating(summary.active_jobs, job.project_name, job.progress)
            } else if summary.queued_jobs > 0 {
                activity = .queued(summary.queued_jobs)
            } else {
                activity = .idle
            }
        } catch {
            activity = .idle
            currentJobID = nil
        }
    }

    private func handleTermination(exitCode: Int32) {
        appendLauncherLog("后端退出，code=\(exitCode)")
        process = nil
        closeLogHandle()
        try? fileManager.removeItem(at: pidURL)

        launchStartedAt = nil
        if requestedStop || exitCode == 0 {
            cleanupAfterStop()
        } else {
            let friendlyMessage = BackendFailureMessage.classify(
                logTail: readBackendLogTail(),
                exitCode: exitCode
            )
            status = .failed(friendlyMessage)
            message = friendlyMessage
        }
    }

    private func cleanupAfterStop() {
        process = nil
        requestedStop = false
        launchStartedAt = nil
        closeLogHandle()
        try? fileManager.removeItem(at: pidURL)
        status = .stopped
        activity = .idle
        currentJobID = nil
        message = "服务已停止，模型内存将由系统回收"
        appendLauncherLog("后端已停止")
    }

    private func updateLoadingMessage() {
        guard let launchStartedAt else { return }
        let elapsed = max(0, Int(Date().timeIntervalSince(launchStartedAt)))
        if elapsed < 180 {
            message = "模型加载中，已等待 \(elapsed) 秒，通常需要 1～2 分钟，请勿重复启动"
        } else {
            let minutes = max(1, elapsed / 60)
            message = "模型仍在加载，已等待约 \(minutes) 分钟；如长时间无响应，请打开日志"
        }
    }

    private func readBackendLogTail(maxBytes: UInt64 = 64 * 1024) -> String {
        guard let handle = try? FileHandle(forReadingFrom: backendLogURL) else { return "" }
        defer { try? handle.close() }
        let size = (try? handle.seekToEnd()) ?? 0
        let tailOffset = size > maxBytes ? size - maxBytes : 0
        let offset = min(size, max(backendLogStartOffset, tailOffset))
        try? handle.seek(toOffset: offset)
        let data = (try? handle.readToEnd()) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    private func scheduleForceStop(pid: pid_t) {
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(6))
            guard let self, self.processExists(pid) else {
                self?.cleanupAfterStop()
                return
            }
            self.appendLauncherLog("正常停止超时，发送 SIGKILL，pid=\(pid)")
            Darwin.kill(pid, SIGKILL)
            try? await Task.sleep(for: .milliseconds(400))
            self.cleanupAfterStop()
        }
    }

    private func prepareDirectories() {
        try? fileManager.createDirectory(at: appSupportURL, withIntermediateDirectories: true)
        try? fileManager.createDirectory(at: logsURL, withIntermediateDirectories: true)
    }

    private func appendLauncherLog(_ line: String) {
        prepareDirectories()
        let formatter = ISO8601DateFormatter()
        let entry = "[\(formatter.string(from: Date()))] \(line)\n"
        let data = Data(entry.utf8)
        if !fileManager.fileExists(atPath: launcherLogURL.path) {
            fileManager.createFile(atPath: launcherLogURL.path, contents: data)
            return
        }
        guard let handle = try? FileHandle(forWritingTo: launcherLogURL) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
    }

    private func storedPID() -> pid_t? {
        guard let text = try? String(contentsOf: pidURL, encoding: .utf8),
              let value = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return nil
        }
        return value
    }

    private func ownedProcessIsRunning() -> Bool {
        process?.isRunning == true
    }

    private func storedProcessIsRunning() -> Bool {
        guard let pid = storedPID() else { return false }
        return processExists(pid)
    }

    private func processExists(_ pid: pid_t) -> Bool {
        guard pid > 1 else { return false }
        return Darwin.kill(pid, 0) == 0 || errno == EPERM
    }

    private func closeLogHandle() {
        try? logHandle?.synchronize()
        try? logHandle?.close()
        logHandle = nil
    }
}
