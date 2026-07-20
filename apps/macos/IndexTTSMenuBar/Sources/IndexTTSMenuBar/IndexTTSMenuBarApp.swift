import AppKit
import IndexTTSMenuBarCore
import SwiftUI

@main
struct IndexTTSMenuBarApp: App {
    @StateObject private var controller = BackendController()

    var body: some Scene {
        MenuBarExtra {
            Text("IndexTTS：\(controller.status.title)")
                .font(.headline)
            if isServiceAvailable(controller.status) {
                Text(controller.activity.title)
                    .font(.caption)
            }

            if !controller.message.isEmpty {
                Text(controller.message)
                    .font(.caption)
            }

            Divider()

            Button("打开生成界面") {
                controller.openWebUI()
            }
            .disabled(!isServiceAvailable(controller.status))

            Button("打开任务中心") {
                controller.openTaskCenter()
            }
            .disabled(!isServiceAvailable(controller.status))

            if controller.currentJobID != nil {
                Button("取消当前任务") {
                    controller.cancelCurrentTask()
                }
            }

            if controller.status == .stopped || isFailure(controller.status) {
                Button("启动服务") {
                    controller.start()
                }
            } else if controller.status == .external {
                Text("当前服务由终端启动，菜单栏不会强行结束它。")
                    .font(.caption)
            } else {
                Button("停止服务并释放内存") {
                    controller.stop()
                }
                .disabled(controller.status == .starting || controller.status == .stopping)

                Button("重新启动服务") {
                    controller.restart()
                }
                .disabled(controller.status == .starting || controller.status == .stopping)
            }

            Divider()

            Button("打开输出文件夹") {
                controller.openOutputs()
            }

            Button("打开日志文件夹") {
                controller.openLogs()
            }

            Button("复制诊断信息") {
                controller.copyDiagnosticInfo()
            }

            Divider()

            Toggle("允许局域网访问（重启后生效）", isOn: $controller.allowLAN)
            Toggle("打开 APP 后自动启动服务", isOn: $controller.startBackendWithApp)
            Toggle(
                "登录后启动菜单栏",
                isOn: Binding(
                    get: { controller.launchAtLogin },
                    set: { controller.setLaunchAtLogin($0) }
                )
            )

            Divider()

            Button("退出菜单栏（保留服务）") {
                controller.quit(stopBackend: false)
            }

            if controller.status != .external {
                Button("退出并停止服务") {
                    controller.quit(stopBackend: true)
                }
            }
        } label: {
            Image(nsImage: StatusBarIconRenderer.image(for: controller.status, activity: controller.activity))
            .interpolation(.high)
            .accessibilityLabel("IndexTTS：\(controller.status.title)，\(controller.activity.title)")
        }
        .menuBarExtraStyle(.menu)
    }

    private func isServiceAvailable(_ status: BackendController.Status) -> Bool {
        status == .running || status == .external
    }

    private func isFailure(_ status: BackendController.Status) -> Bool {
        if case .failed = status { return true }
        return false
    }
}
